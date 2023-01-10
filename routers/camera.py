import models
import utils

from logging import getLogger
from fastapi import APIRouter
from multiprocessing import Process
from fastapi.responses import StreamingResponse

router = APIRouter()
logger = getLogger("app")

# Init controler
face_tracker_controler = models.FaceTracker()

p = Process(target=face_tracker_controler.run)
p.start()

logger.info("Init FaceTracker Successfull")

@router.get("/streaming")
async def predict(is_default: bool = False):
 return StreamingResponse(
     utils.gen_frame(face_tracker_controler, is_default),
     media_type="multipart/x-mixed-replace; boundary=frame",
 )
