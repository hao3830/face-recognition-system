import utils
import models

import cv2
import time
import depthai as dai

from multiprocessing import Queue, Manager, Process
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
            data["is_kill"] = False

            p1 = Thread(target=self.send_reg_api, args=(Q, data))
            p1.start()

            p2 = Thread(target=self.convert_frame, args=(data,))
            p2.start()

            pipeline = models.get_pipeline(self.manager["det_conf"])

            # Pipeline defined, now the device is connected to
            self.device = dai.Device(pipeline, usb2Mode=True)

            # Start the pipeline
            self.device.startPipeline()

            preview = self.device.getOutputQueue("preview", maxSize=30, blocking=False)
            tracklets = self.device.getOutputQueue(
                "tracklets", maxSize=30, blocking=False
            )

            startTime = time.monotonic()
            counter = 0
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

                frame = imgFrame.getCvFrame()
                # frame = cv2.resize(frame,(500,500))
                self.manager["image_size"] = frame.shape
                new_frame = frame.copy()
                default_frame = frame.copy()
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
                    #TODO: show result face detected but not comfortable
                    if not self.check_face_size(bbox):
                        continue
                    
                    if (
                        str(t.id) in data
                        and data[str(t.id)]["face_quality_valid"] == False
                        and not data[str(t.id)]["face_quality_valid"]
                        and (time.time() - data[str(t.id)]["last_check_time"])
                        >= self.manager["check_freq"] 
                        and data[str(t.id)]['sent'] < self.manager["max_time_check"]
                        and not data[str(t.id)]["start_sent"]
                        and t.status == dai.Tracklet.TrackingStatus.TRACKED
                        and isContain == True
                    ):
                        Q.put((default_frame, bbox, str(t.id)))

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
                            utils.ImageProcess.blue,
                        )
                        cv2.putText(
                            frame,
                            STATUS_MAP[t.status],
                            (x1 + 10, y1 + 50),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.5,
                            utils.ImageProcess.blue,
                        )

                        if str(t.id) in data and data[str(t.id)]['face_quality_valid']:
                            box_color = utils.ImageProcess.green
                        elif str(t.id) in data and data[str(t.id)]['sent'] >= self.manager["max_time_check"]:
                            box_color = utils.ImageProcess.red
                        elif str(t.id) in data and data[str(t.id)]['start_sent']:
                            box_color = utils.ImageProcess.yellow
                        else:
                            box_color = utils.ImageProcess.light_grey
        
                        frame = utils.ImageProcess.draw_4_rounded_conner_bbox(
                            frame, bbox, box_color, thickness=3
                        )
                        new_frame = utils.ImageProcess.draw_4_rounded_conner_bbox(
                            new_frame, bbox, box_color, thickness=3
                        )

                    # Tracking save
                    if t.status == dai.Tracklet.TrackingStatus.NEW:
                        data[str(t.id)] = {
                            "face_quality_valid": False,
                            "sent": 0,
                            "last_check_time": time.time(),
                            'start_sent': False,
                        }  # Reset
                    elif (
                        t.status == dai.Tracklet.TrackingStatus.TRACKED
                        and str(t.id) in data
                    ):
                        #                data[str(t.id)]['lostCnt'] = 0
                        data[str(t.id)] = {**data[str(t.id)], "lostCnt": 0}
                    elif (
                        t.status == dai.Tracklet.TrackingStatus.LOST
                        and str(t.id) in data
                    ):
                        curr = data[str(t.id)]
                        if "lostCnt" in curr:
                            curr["lostCnt"] += 1
                        else:
                            curr["lostCnt"] = 0

                        data[str(t.id)] = {**curr}
                        # If tracklet has been "LOST" for more than 10 frames, remove it
                        if (
                            str(t.id) in data
                            and "lostCnt" in data[str(t.id)]
                            and 10 < data[str(t.id)]["lostCnt"]
                        ):
                            data.pop(str(t.id))
                    elif (t.status == dai.Tracklet.TrackingStatus.REMOVED) and str(
                        t.id
                    ) in data:
                        data.pop(str(t.id))

                cv2.putText(
                    frame,
                    "FPS: {:.2f}".format(fps),
                    (2, 10),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    utils.ImageProcess.blue,
                )

                self.manager["frame"] = frame
                self.manager["frame_default"] = new_frame

                if self.manager["is_restart"]:
                    data["is_kill"] = True
                    return
        except Exception as error:

            data["is_kill"] = True
            # time.sleep(0.5)
            p1.join()
            del p1

            p2.join()
            del p2

            Q.close()
            Q.join_thread()
            del Q
            del data
            raise (error)

    def convert_frame(self, data):
        while True:
            # print("1")
            if data["is_kill"]:
                break

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

        send_api_counter = 0

        while True:
            if data["is_kill"]:
                break

            if Q.empty():
                continue
            FRAME, BBOX, idx = Q.get()
            cropped = FRAME[BBOX[1] : BBOX[3], BBOX[0] : BBOX[2]]
            cropped_bytes = cv2.imencode(".jpg", cropped)[1].tobytes()
            import time
            start = time.time()
            status = models.face_quality.run(cropped)
            print(time.time() - start)
            if status == "bad":
                continue
            if str(idx) not in data:
                continue
            curr = data[str(idx)]
            curr['start_sent'] = True
            data[str(idx)] = {**curr}

            logger.info("Sent Face Image")
            headers = {"Authorization": f"Bearer {self.TOKEN}"}

            if status != "bad":
                if str(idx) in data:
                    data[str(idx)] = {**curr}
                respone = utils.search_face(cropped_bytes=cropped_bytes, headers=headers)
                respone = respone.json()
                

                if respone['str_code'] == 'NotFound':
                    curr["face_quality_valid"] = False
                elif respone['str_code'] == 'Done':
                    _ = utils.insert_face(cropped_bytes=cropped_bytes, headers=headers)
                    curr["face_quality_valid"] = True
                
                curr['sent'] += 1

                if curr['sent'] >= self.manager["max_time_check"]:
                    _ = utils.insert_face(cropped_bytes=cropped_bytes, headers=headers)
                    curr['sent'] = 0


            curr["last_check_time"] = time.time()
            curr['start_sent'] = False
            if str(idx) in data:
                data[str(idx)] = {**curr}


            if send_api_counter % 1000 == 0:
                self.TOKEN = utils.get_token()

            send_api_counter += 1

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
            "conf": self.manager["det_conf"],
            "max_time_check": self.manager["max_time_check"],
            "check_freq": self.manager["check_freq"],
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

    def check_face_size(self, bbox):
        if (
            bbox[2] - bbox[0] < self.size_face_manager["w"]
            or bbox[3] - bbox[1] < self.size_face_manager["h"]
        ):
            return False
        return True
    
    def get_models_settings(self):
        pass