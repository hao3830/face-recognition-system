

from fastapi import APIRouter

router = APIRouter()

@router.get("/configs")
def get_config():
    pass

@router.post("/configs")
def set_config():
    pass