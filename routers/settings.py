import models
import utils

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


class DetectionSettings(BaseModel):
    conf: float
    max_time_check: int
    check_freq: int


@router.get("/settings")
def get_settings():
    roi = models.face_tracker_controler.get_roi()
    face_settings = models.face_tracker_controler.get_size_face()
    det_settings = models.face_tracker_controler.get_det_settings()

    return {
        "code": 200,
        "message": "200OK",
        "str_code": "200 OK",
        "roi": roi,
        "face_settings": face_settings,
        "det_settings": det_settings,
    }


@router.post("/settings")
def set_settings(roi: ROI, face_size: FaceSize, det_settings: DetectionSettings):
    models.face_tracker_controler.set_roi(roi.x, roi.y, roi.w, roi.h)
    models.face_tracker_controler.set_size_face(face_size.w, face_size.h)

    data = {
        "conf": det_settings.conf,
        "max_time_check": det_settings.max_time_check,
        "check_freq": det_settings.check_freq,
    }

    models.face_tracker_controler.set_det_settings(data)
    
    data = utils.get_settings_config_without_parse_yaml()
    # size face settings
    data["size_face_w"] = face_size.w
    data["size_face_w"] = face_size.h

    # Roi settings
    data["roi"]["x"] = roi.x
    data["roi"]["y"] = roi.y
    data["roi"]["w"] = roi.w
    data["roi"]["h"] = roi.h

    # detection settings
    data["det_conf"] = det_settings.conf
    data["check_freq"] = det_settings.check_freq
    data["max_time_check"] = det_settings.max_time_check

    result = utils.set_settings_config(data)

    if result:
        models.face_tracker_controler.set_check_restart()
        return {"code": 200, "message": "200OK", "str_code": "200 OK"}

    return {
        "code": 400,
        "message": "InternalServerError",
        "str_code": "Internal Server Error",
    }


@router.get("/image_size")
def get_image_size():
    image_size = models.face_tracker_controler.get_image_size()
    if image_size is not None:
        return {
            "code": 200,
            "message": "200OK",
            "str_code": "200 OK",
            "image_size": image_size,
        }

@router.get("/face_quality_assessment")
def get_face_quality_assessment():

    data = models.face_quality.get_face_quality_assessment()
    return_info = {
        "min_detection_confidence": data['min_detection_confidence'],
        "face_angle": data['face_angle'],
        "ear": data['ear'],
    }

    return {
        "code": 200,
        "message": "200OK",
        "str_code": "200 OK",
        "face_quality_assessment": return_info
    }

class FaceQualityAssessment(BaseModel):
    min_detection_confidence: float
    face_angle: float
    ear: float

@router.post("/face_quality_assessment")
def set_face_quality_assessment(face_quality_assessment: FaceQualityAssessment):
    data = {
        "min_detection_confidence": face_quality_assessment.min_detection_confidence,
        "face_angle": face_quality_assessment.face_angle,
        "ear": face_quality_assessment.ear,
    }

    result = utils.set_face_quality_assessment_config(data)
    if result:
        # models.face_tracker_controler.set_check_restart()
        return {"code": 200, "message": "200OK", "str_code": "200 OK"}

    return {
        "code": 400,
        "message": "InternalServerError",
        "str_code": "Internal Server Error",
    }
