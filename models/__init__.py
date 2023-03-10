import time
import utils

from threading import Thread

from models.face_tracker import FaceTracker
from models.depthai import get_pipeline
# from models.face_quality import FaceQuality
from models.face_quality_2 import FaceQuality

# face_quality_configs = utils.get_face_quality_config()
face_quality_assessment = utils.get_face_quality_assessment_config()
# face_quality = FaceQuality(
#     face_quality_configs.backbone_path,
#     face_quality_configs.quality_path,
#     face_quality_configs.confident,
# )
face_quality = FaceQuality(
    min_detection_confidence=face_quality_assessment.min_detection_confidence,
    threshold_angle=face_quality_assessment.face_angle,
    close_eye_thres=face_quality_assessment.ear
)

face_tracker_controler = FaceTracker()


def auto_restart():
    global face_quality
    # global face_tracker_controler
    while True:
        try:
            face_tracker_controler.run()
        except Exception as err:
            print(f"err: {err}")
            # del face_tracker_controler
            
            #Redefine 
            # face_tracker_controler = FaceTracker()
        del face_quality
        face_quality = FaceQuality(
            min_detection_confidence=face_quality_assessment.min_detection_confidence,
            threshold_angle=face_quality_assessment.face_angle,
            close_eye_thres=face_quality_assessment.ear
        )

        if face_tracker_controler.device is not None:
            face_tracker_controler.device.close()

        time.sleep(0.5)


thread = Thread(target=auto_restart)
thread.start()
