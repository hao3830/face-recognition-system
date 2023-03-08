import time
import utils

from threading import Thread

from models.face_tracker import FaceTracker
from models.depthai import get_pipeline
# from models.face_quality import FaceQuality
from models.face_quality_2 import FaceQuality

face_quality_configs = utils.get_face_quality_config()

# face_quality = FaceQuality(
#     face_quality_configs.backbone_path,
#     face_quality_configs.quality_path,
#     face_quality_configs.confident,
# )
face_quality = FaceQuality()

face_tracker_controler = FaceTracker()


def auto_restart():
    # global face_tracker_controler
    while True:
        try:
            face_tracker_controler.run()
        except Exception as err:
            print(f"err: {err}")
            # del face_tracker_controler
            
            #Redefine 
            # face_tracker_controler = FaceTracker()

        if face_tracker_controler.device is not None:
            face_tracker_controler.device.close()

        time.sleep(0.5)


thread = Thread(target=auto_restart)
thread.start()
