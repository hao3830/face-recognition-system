import utils

import cv2
import time
import requests
import blobconverter
import depthai as dai

from multiprocessing import Process, Queue, Manager, Value
from logging import getLogger

logger = getLogger("app")


class FaceTracker:
    def __init__(self):
        self.frame = Queue()
        self.frame_default = Queue()
        self.drawed_frame_buffer_queue = None
        self.default_frame_buffer_queue = None
        self.TOKEN = utils.get_token()
        self.manager = Manager().dict()

    def run(self):

        statusMap = {
            dai.Tracklet.TrackingStatus.NEW: "NEW",
            dai.Tracklet.TrackingStatus.TRACKED: "TRACKED",
            dai.Tracklet.TrackingStatus.LOST: "LOST",
            dai.Tracklet.TrackingStatus.REMOVED: "REMOVED",
        }

        Q = Queue()
        data = Manager().dict()

        p = Process(target=self.send_reg_api, args=(Q, data))
        p.start()

        p = Process(target=self.convert_frame, args=())
        p.start()

        # Start defining a pipeline
        pipeline = dai.Pipeline()

        colorCam = pipeline.createColorCamera()
        detectionNetwork = pipeline.createMobileNetDetectionNetwork()
        objectTracker = pipeline.createObjectTracker()
        trackerOut = pipeline.createXLinkOut()

        xlinkOut = pipeline.createXLinkOut()

        xlinkOut.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        colorCam.setPreviewSize(720, 720)
        colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        colorCam.setInterleaved(False)
        colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        # colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
        colorCam.setFps(40)

        # setting node configs
        detectionNetwork.setBlobPath(
            blobconverter.from_zoo(name="face-detection-retail-0005", shaves=6)
        )
        detectionNetwork.setConfidenceThreshold(0.5)
        detectionNetwork.input.setBlocking(False)

        face_det_manip = pipeline.create(dai.node.ImageManip)
        face_det_manip.initialConfig.setResize(300, 300)
        face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
        colorCam.preview.link(face_det_manip.inputImage)
        face_det_manip.out.link(detectionNetwork.input)

        # Link plugins CAM . NN . XLINK
        # colorCam.preview.link(detectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

        # objectTracker.setDetectionLabelsToTrack([15])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(
            dai.TrackerIdAssignmentPolicy.SMALLEST_ID
        )

        # if fullFrameTracking:
        colorCam.preview.link(objectTracker.inputTrackerFrame)
        # else:
        #    detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        detectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.out.link(trackerOut.input)

        # Pipeline defined, now the device is connected to
        with dai.Device(pipeline) as device:

            # Start the pipeline
            device.startPipeline()

            preview = device.getOutputQueue("preview")
            tracklets = device.getOutputQueue("tracklets")

            startTime = time.monotonic()
            counter = 0
            fps = 0
            frame = None
            prev_status = None

            while True:
                imgFrame = preview.get()
                track = tracklets.get()

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

                trackletsData = track.tracklets
                for t in trackletsData:
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)
                    bbox = [x1, y1, x2, y2]
                    if (
                        str(t.id) in data
                        and data[str(t.id)]["bad"] == False
                        and t.status == dai.Tracklet.TrackingStatus.TRACKED
                        and data[str(t.id)]["sent"] < TIME_TRACK
                        and counter % SKIP_TRACK_TIME == 0
                    ):

                        Q.put((new_frame, bbox, str(t.id)))
                    if (
                        statusMap[t.status] != "LOST"
                        and statusMap[t.status] != "REMOVED"
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
                            statusMap[t.status],
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
        print("GO")
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

            if str(idx) not in data:
                continue
            if data[str(idx)]["sent"] >= TIME_TRACK or data[str(idx)]["bad"] == True:
                continue
            cropped = FRAME[BBOX[1] : BBOX[3], BBOX[0] : BBOX[2]]
            cropped_bytes = cv2.imencode(".jpg", cropped)[1].tobytes()
            status = utils.good_bad_face(cropped_bytes)

            try:
                curr = data[str(idx)]
                if status == "bad" and curr["sent"] < TIME_TRACK:
                    continue
                logger.info("Sent Face Image")
                headers = {"Authorization": f"Bearer {self.TOKEN}"}

                res = requests.post(
                    url=REG_API_CHECK, files=dict(file=cropped_bytes), headers=headers
                )
                res = res.json()

                if res["code"] == 1000:
                    curr["bad"] = True
                    res = requests.post(
                        url=INSERT_API, files=dict(file=cropped_bytes), headers=headers
                    )

                curr["sent"] += 1
                data[str(idx)] = {**curr}
                if (
                    data[str(idx)]["sent"] == TIME_TRACK
                    and data[str(idx)]["bad"] == False
                ):
                    res = requests.post(
                        url=INSERT_API, files=dict(file=cropped_bytes), headers=headers
                    )

            except:
                logger.error("Fail connect to insert face api")
