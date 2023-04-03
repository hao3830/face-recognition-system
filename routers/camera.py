import models
import utils

from fastapi import APIRouter
from logging import getLogger
from fastapi.responses import StreamingResponse, Response

router = APIRouter()
logger = getLogger("app")

@router.get("/streaming")
async def predict(is_default: bool = False):
    return StreamingResponse(
        utils.gen_frame(models.face_tracker_controler, is_default),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )