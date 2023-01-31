import time

from threading import Thread

from models.face_tracker import FaceTracker
from models.depthai import get_pipeline

face_tracker_controler = FaceTracker()

def auto_restart():
    while True:
        try:
            face_tracker_controler.run()
        except Exception as err:
            print(f"err: {err}")
        if face_tracker_controler.device is not None:
            face_tracker_controler.device.close()
        
        time.sleep(0.5)
thread = Thread(target=auto_restart)
thread.start()