from threading import Thread

from models.face_tracker import FaceTracker
from models.depthai import get_pipeline


# Init controler
face_tracker_controler = FaceTracker()

thread = Thread(target=face_tracker_controler.run)
thread.start()