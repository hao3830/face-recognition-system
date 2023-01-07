import models
import utils

from threading import Thread
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from logging import getLogger

router = APIRouter()
logger = getLogger("app")

# Init controler
face_tracker_controler = models.FaceTracker()

thread = Thread(target=face_tracker_controler.run)
thread.start()
logger.info("Init FaceTracker Successfull")

@router.get("/streaming")
async def predict(is_default: bool = False):
 return StreamingResponse(
     utils.gen_frame(face_tracker_controler, is_default),
     media_type="multipart/x-mixed-replace; boundary=frame",
 )
