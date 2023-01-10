import models

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()

@router.get("/roi_settings")
def get_roi_settings():
    return models.face_tracker_controler.get_roi()

@router.get("/face_size_settings")
def get_face_size_settings():
    return models.face_tracker_controler.get_size_face()

class ROI(BaseModel):
    x: int
    y: int
    w: int
    h: int

@router.post("/roi_settings")
def set_roi_settings(roi: ROI):
    models.face_tracker_controler.set_roi(roi.x, roi.y, roi.w, roi.h)
    return "OK"

class FaceSize(BaseModel):
    w: int
    h: int

@router.post("/face_size_settings")
def set_face_size_settings(face_size: FaceSize):
    models.face_tracker_controler.set_size_face(face_size.w, face_size.h)
    return "OK"