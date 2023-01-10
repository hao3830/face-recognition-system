import blobconverter


import depthai as dai


def get_pipeline():
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
    colorCam.setFps(30)

    # setting node configs
    detectionNetwork.setBlobPath(
        blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6)
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

    colorCam.preview.link(objectTracker.inputTrackerFrame)

    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)
    objectTracker.out.link(trackerOut.input)

    return pipeline
