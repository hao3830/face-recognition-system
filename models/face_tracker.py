import utils
import models

import cv2
import time
import depthai as dai

from multiprocessing import Queue, Manager
from threading import Thread
from logging import getLogger

logger = getLogger("app")

settings = utils.get_settings_config()



STATUS_MAP = {
    dai.Tracklet.TrackingStatus.NEW: "NEW",
    dai.Tracklet.TrackingStatus.TRACKED: "TRACKED",
    dai.Tracklet.TrackingStatus.LOST: "LOST",
    dai.Tracklet.TrackingStatus.REMOVED: "REMOVED",
}


class FaceTracker:
    def __init__(self):
        self.frame = Queue()
        self.frame_default = Queue()
        self.drawed_frame_buffer_queue = None
        self.default_frame_buffer_queue = None
        self.manager = Manager().dict()

        self.manager["drawed_frame_buffer"] = None
        self.manager["default_frame_buffer"] = None
        self.manager["is_restart"] = False
        self.manager["det_conf"] = settings.det_conf
        self.manager["image_size"] = None

        self.manager["max_time_check"] = settings.max_time_check
        self.manager["check_freq"] = settings.check_freq

        self.TOKEN = utils.get_token()

        # Size face settings
        size_face_w = settings.size_face_w
        size_face_h = settings.size_face_h
        self.size_face_manager = Manager().dict()
        self.size_face_manager["w"] = size_face_w
        self.size_face_manager["h"] = size_face_h

        # ROI settings
        self.roi_manager = Manager().dict()
        roi = settings.roi
        self.roi_manager["x"] = roi.x
        self.roi_manager["y"] = roi.y
        self.roi_manager["w"] = roi.w
        self.roi_manager["h"] = roi.h
        
        self.device = None


    def run(self):
        try:
            Q = Queue(maxsize=30)
            data = Manager().dict()

            p1 = Thread(target=self.send_reg_api, args=(Q, data))
            p1.start()

            p2 = Thread(target=self.convert_frame, args=(data,))
            p2.start()

            pipeline = models.get_pipeline(self.manager["det_conf"])

            # Pipeline defined, now the device is connected to
            self.device = dai.Device(pipeline, usb2Mode=True)
        except Exception as error:
            data['is_kill'] = True
            p1.join()
            p2.join()
            raise(error)
        

        # Start the pipeline
        self.device.startPipeline()

        preview = self.device.getOutputQueue("preview", maxSize=30, blocking=False)
        tracklets = self.device.getOutputQueue("tracklets", maxSize=30, blocking=False)

        startTime = time.monotonic()
        counter = 0
        check_frequent_counter = 0
        fps = 0
        frame = None
        while True:

            self.manager["is_restart"] = False

            imgFrame = preview.get()
            track = tracklets.get()
            if imgFrame is None or track is None:
                continue

            counter += 1
            current_time = time.monotonic()
            if (current_time - startTime) > 1:
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            color = (255, 0, 0)
            frame = imgFrame.getCvFrame()
            # frame = cv2.resize(frame,(500,500))
            self.manager["image_size"] = frame.shape
            new_frame = frame.copy()
            overlay = frame.copy()

            trackletsData = track.tracklets

            limit_roi = [
                self.roi_manager["x"],
                self.roi_manager["y"],
                self.roi_manager["x"] + self.roi_manager["w"],
                self.roi_manager["y"] + self.roi_manager["h"],
            ]

            cv2.rectangle(
                overlay,
                (limit_roi[0], limit_roi[1]),
                (limit_roi[2], limit_roi[3]),
                (0, 200, 0),
                -1,
            )

            alpha = 0.4  

            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            for t in trackletsData:
                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                bbox = [x1, y1, x2, y2]

                isContain = utils.ImageProcess.isContain(bbox, limit_roi)

                if (
                    str(t.id) in data
                    and data[str(t.id)]["face_quality_valid"] == False
                    and t.status == dai.Tracklet.TrackingStatus.TRACKED
                    # and data[str(t.id)]["sent"] < self.manager["max_time_check"]
                    and check_frequent_counter % self.manager["check_freq"] == 0
                    and isContain
                ):
                    Q.put((new_frame, bbox, str(t.id)))
                check_frequent_counter += 1
                if (
                    STATUS_MAP[t.status] != "LOST"
                    and STATUS_MAP[t.status] != "REMOVED"
                ):
                    cv2.putText(
                        frame,
                        f"ID:{t.id}",
                        (x1 + 10, y1 + 35),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        color,
                    )
                    cv2.putText(
                        frame,
                        STATUS_MAP[t.status],
                        (x1 + 10, y1 + 50),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        color,
                    )

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX
                    )

                # Tracking save
                if t.status == dai.Tracklet.TrackingStatus.NEW:
                    data[str(t.id)] = {"face_quality_valid": False, "sent": 0}  # Reset
                elif t.status == dai.Tracklet.TrackingStatus.TRACKED:
                    #                data[str(t.id)]['lostCnt'] = 0
                    data[str(t.id)] = {**data[str(t.id)], "lostCnt": 0}
                elif t.status == dai.Tracklet.TrackingStatus.LOST:
                    curr = data[str(t.id)]
                    curr["lostCnt"] += 1
                    data[str(t.id)] = {**curr}
                    # If tracklet has been "LOST" for more than 10 frames, remove it
                    if (
                        10 < data[str(t.id)]["lostCnt"]
                        and "lost" not in data[str(t.id)]
                    ):
                        curr = data[str(t.id)]
                        curr["lost"] = True
                        data[str(t.id)] = {**curr}
                elif (
                    t.status == dai.Tracklet.TrackingStatus.REMOVED
                ) and "lost" not in data[str(t.id)]:
                    data.pop(str(t.id))

            cv2.putText(
                frame,
                "FPS: {:.2f}".format(fps),
                (2, 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )

            self.manager["frame"] = frame
            self.manager["frame_default"] = new_frame

            if self.manager["is_restart"]:
                return

    def convert_frame(self, data):
        while True:
            
            if 'is_kill' in data:
                return

            if "frame" not in self.manager or "frame_default" not in self.manager:
                continue
            frame = self.manager["frame"]
            frame_default = self.manager["frame_default"]
            if frame is None or frame_default is None:
                continue

            self.manager["drawed_frame_buffer"] = cv2.imencode(".jpg", frame)[
                1
            ].tobytes()
            self.manager["default_frame_buffer"] = cv2.imencode(".jpg", frame_default)[
                1
            ].tobytes()

    def get(self):
        return self.manager["drawed_frame_buffer"]

    def get_default(self):
        return self.manager["default_frame_buffer"]

    def send_reg_api(self, Q, data):

        counter = 0

        while True:
            if 'is_kill' in data:
                return

            if Q.empty():
                continue
            FRAME, BBOX, idx = Q.get()

            # Check face size
            if (
                BBOX[2] - BBOX[0] < self.size_face_manager["w"]
                or BBOX[3] - BBOX[1] < self.size_face_manager["h"]
            ):
                continue
                
            if str(idx) not in data:
                continue
            
            # IS FACE IMAGE HAS SENT => RESET
            if (
                data[str(idx)]["sent"] >= self.manager["max_time_check"]
                or data[str(idx)]["face_quality_valid"] == True
            ):
                continue

            cropped = FRAME[BBOX[1] : BBOX[3], BBOX[0] : BBOX[2]]
            cropped_bytes = cv2.imencode(".jpg", cropped)[1].tobytes()
            status = utils.good_bad_face(cropped_bytes)

            if str(idx) not in data:
                continue
            curr = data[str(idx)]
            if status == "bad" and curr["sent"] < self.manager["max_time_check"]:
                continue
            logger.info("Sent Face Image")
            headers = {"Authorization": f"Bearer {self.TOKEN}"}

            if status != "bad":
                curr["face_quality_valid"] = True
                data[str(idx)] = {**curr}
                res = utils.insert_face(cropped_bytes=cropped_bytes, headers=headers)
                if res is None:
                    res = utils.insert_face(
                        cropped_bytes=cropped_bytes, headers=headers
                    )
                    if res is None:
                        continue
            
            if counter % 1000 == 0:
                self.TOKEN = utils.get_token()

            counter += 1
            #CHECK IS FACE VALID BEFORE POST TO THE SERVER

            # res = utils.search_face(cropped_bytes=cropped_bytes, headers=headers)
            # if res is None:
            #     self.TOKEN = utils.get_token()
            #     res = utils.search_face(cropped_bytes=cropped_bytes, headers=headers)
            #     if res is None:
            #         continue
            # res = res.json()

            # if res["code"] == 1000:
            #     curr["face_quality_valid"] = True
            #     res = utils.insert_face(cropped_bytes=cropped_bytes, headers=headers)
            #     if res is None:
            #         self.TOKEN = utils.get_token()
            #         res = utils.insert_face(
            #             cropped_bytes=cropped_bytes, headers=headers
            #         )
            #         if res is None:
            #             continue
                
            

            # if str(idx) not in data:
            #     continue
            # curr["sent"] += 1
            # data[str(idx)] = {**curr}
            # if (
            #     data[str(idx)]["sent"] == self.manager["max_time_check"]
            #     and data[str(idx)]["face_quality_valid"] == False
            # ):
            #     res = utils.insert_face(cropped_bytes=cropped_bytes, headers=headers)
            #     if res is None:
            #         res = utils.insert_face(
            #             cropped_bytes=cropped_bytes, headers=headers
            #         )
            #         if res is None:
            #             continue
            
            # REFRESH TOKEN
            

    def get_roi(self):
        return self.roi_manager

    def get_size_face(self):
        return self.size_face_manager

    def set_roi(self, x, y, w, h):
        self.roi_manager["x"] = x
        self.roi_manager["y"] = y
        self.roi_manager["w"] = w
        self.roi_manager["h"] = h

    def set_size_face(self, w, h):
        self.size_face_manager["w"] = w
        self.size_face_manager["h"] = h
    
    def get_det_settings(self):
        data = {
            'conf': self.manager["det_conf"],
            'max_time_check': self.manager["max_time_check"],
            'check_freq': self.manager["check_freq"]
        }
        return data
    
    def set_det_settings(self, data):
        self.manager["det_conf"] = data["conf"]
        self.manager["max_time_check"] = data["max_time_check"]
        self.manager["check_freq"] = data["check_freq"]

    def set_check_restart(self):
        self.manager["is_restart"] = True
    
    def get_image_size(self):

        if self.manager["image_size"] is not None:
            height, width, _ = self.manager["image_size"]
            return {
                "height": height,
                "width": width,
            }

        return None