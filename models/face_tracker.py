import utils
import models

import cv2
import time
import depthai as dai

from multiprocessing import Process, Queue, Manager
from logging import getLogger

logger = getLogger("app")

settings = utils.get_settings_config()

MAX_TIME_CHECK = settings.max_time_check
CHECK_FREQ = settings.check_freq

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

    def run(self):

        Q = Queue()
        data = Manager().dict()

        p = Process(target=self.send_reg_api, args=(Q, data))
        p.start()

        p = Process(target=self.convert_frame, args=())
        p.start()

        pipeline = models.get_pipeline()

        # Pipeline defined, now the device is connected to
        with dai.Device(pipeline) as device:

            # Start the pipeline
            device.startPipeline()

            preview = device.getOutputQueue("preview", maxSize=30, blocking=False)
            tracklets = device.getOutputQueue("tracklets", maxSize=30, blocking=False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            frame = None
            while True:
                imgFrame = preview.tryGet()
                track = tracklets.tryGet()
                if imgFrame is None or track is None:
                    continue

                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                color = (0, 255, 0)
                frame = imgFrame.getCvFrame()
                # frame = cv2.resize(frame,(500,500))
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

                alpha = 0.4  # Transparency factor.

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
                        and data[str(t.id)]["bad"] == False
                        and t.status == dai.Tracklet.TrackingStatus.TRACKED
                        and data[str(t.id)]["sent"] < MAX_TIME_CHECK
                        and counter % CHECK_FREQ == 0
                        and isContain
                    ):

                        Q.put((new_frame, bbox, str(t.id)))
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
                        data[str(t.id)] = {"bad": False, "sent": 0}  # Reset
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

    def convert_frame(self):
        while True:
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
        while True:
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
            if (
                data[str(idx)]["sent"] >= MAX_TIME_CHECK
                or data[str(idx)]["bad"] == True
            ):
                continue
            cropped = FRAME[BBOX[1] : BBOX[3], BBOX[0] : BBOX[2]]
            cropped_bytes = cv2.imencode(".jpg", cropped)[1].tobytes()
            status = utils.good_bad_face(cropped_bytes)

            curr = data[str(idx)]
            if status == "bad" and curr["sent"] < MAX_TIME_CHECK:
                continue
            logger.info("Sent Face Image")
            headers = {"Authorization": f"Bearer {self.TOKEN}"}

            res = utils.search_face(cropped_bytes=cropped_bytes, headers=headers)
            if res is None:
                self.TOKEN = utils.get_token()
                res = utils.search_face(cropped_bytes=cropped_bytes, headers=headers)
                if res is None:
                    continue
            res = res.json()

            if res["code"] == 1000:
                curr["bad"] = True
                res = utils.insert_face(cropped_bytes=cropped_bytes, headers=headers)
                if res is None:
                    self.TOKEN = utils.get_token()
                    res = utils.insert_face(
                        cropped_bytes=cropped_bytes, headers=headers
                    )
                    if res is None:
                        continue

            curr["sent"] += 1
            data[str(idx)] = {**curr}
            if (
                data[str(idx)]["sent"] == MAX_TIME_CHECK
                and data[str(idx)]["bad"] == False
            ):
                res = utils.insert_face(cropped_bytes=cropped_bytes, headers=headers)
                if res is None:
                    self.TOKEN = utils.get_token()
                    res = utils.insert_face(
                        cropped_bytes=cropped_bytes, headers=headers
                    )
                    if res is None:
                        continue

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
