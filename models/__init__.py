from multiprocessing import Process
from threading import Thread

from models.face_tracker import FaceTracker
from models.depthai import get_pipeline

face_tracker_controler = FaceTracker()

def auto_restart():
    while 1:
        try:
            face_tracker_controler.run()
        except:
            print("Error")
    
thread = Thread(target=auto_restart)
thread.start()
