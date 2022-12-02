from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import blobconverter
from multiprocessing import Process, Queue, Manager
import requests
import time


REG_API = "http://192.168.20.150:5000/checkin/insert"
statusMap = {dai.Tracklet.TrackingStatus.NEW : "NEW", dai.Tracklet.TrackingStatus.TRACKED : "TRACKED", dai.Tracklet.TrackingStatus.LOST : "LOST",dai.Tracklet.TrackingStatus.REMOVED: "REMOVED"}


Q = Queue()
data = Manager().dict({})
def send_reg_api(Q, data):
    while True:
        if Q.empty() :
            continue
        FRAME, BBOX, idx = Q.get()
        cropped = FRAME[BBOX[1]:BBOX[3],BBOX[0]:BBOX[2]]
        cropped_bytes = cv2.imencode(".jpg",cropped)[1].tobytes()
        res = requests.post(url=REG_API,files=dict(avatar=cropped_bytes))
        
        res = res.json()
        if "bad" in res:
            data[str(idx)]["bad"] = True
        else:
            data[str(idx)]["bad"] = False
        
p = Process(target=send_reg_api,args=(Q,data,))
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

colorCam.setPreviewSize(1080, 1080)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setFps(40)

# setting node configs
detectionNetwork.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0005", shaves=6))
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.input.setBlocking(False)

face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
colorCam.preview.link(face_det_manip.inputImage)
face_det_manip.out.link(detectionNetwork.input)

# Link plugins CAM . NN . XLINK
#colorCam.preview.link(detectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)


#objectTracker.setDetectionLabelsToTrack([15])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

#if fullFrameTracking:
colorCam.preview.link(objectTracker.inputTrackerFrame)
#else:
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


    while(True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        color = (255, 0, 0)
        frame = imgFrame.getCvFrame()
        new_frame = frame.copy()
        trackletsData = track.tracklets
        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)
            bbox = [x1,y1,x2,y2]
            if str(t.id) in data and  ('lostCnt' not in data[str(t.id)] or data[idx]["bad"] ) and t.status == dai.Tracklet.TrackingStatus.TRACKED:
                Q.put((new_frame,bbox,str(t.id)))
            #cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            if statusMap[t.status] != "LOST" and statusMap[t.status] != "REMOVED":
                cv2.putText(frame, statusMap[t.status], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            
            #Tracking save
            if t.status == dai.Tracklet.TrackingStatus.NEW:
                data[str(t.id)] = {} # Reset
                data[idx]["bad"] = False
            elif t.status == dai.Tracklet.TrackingStatus.TRACKED:
                data[str(t.id)]['lostCnt'] = 0
            elif t.status == dai.Tracklet.TrackingStatus.LOST:
                if 'lostCnt' not in data[str(t.id)]:
                    data[str(t.id)]['lostCnt'] = 0
                data[str(t.id)]['lostCnt'] += 1
                # If tracklet has been "LOST" for more than 10 frames, remove it
                if 10 < data[str(t.id)]['lostCnt'] and "lost" not in data[str(t.id)]:
                    #node.warn(f"Tracklet {t.id} lost: {data[str(t.id)]['lostCnt']}")
                    del data[str(t.id)]
                    data[str(t.id)]["lost"] = True
            elif (t.status == dai.Tracklet.TrackingStatus.REMOVED) and "lost" not in data[str(t.id)]:
                del data[str(t.id)]
            
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2,10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    
        cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            break
