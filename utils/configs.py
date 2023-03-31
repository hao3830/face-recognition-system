import yaml
from easydict import EasyDict
from logging import getLogger

logger = getLogger("app")


def parse_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return EasyDict(data)


def _get_config(path):
    config = parse_yaml(path)
    return config

def _get_config_without_parse_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def _set_config(path, data):
    try:
        with open(path, "w") as f:
            yaml.dump(data, f)

        return True
    except Exception as error:
        logger.error(f"error: {error}")
        return False

def get_config():
    return _get_config("configs/app.yaml")

def get_api_config():
    return _get_config("configs/api.yaml")

def get_settings_config():
    return _get_config("configs/settings.yaml")

def get_settings_config_without_parse_yaml():
    return _get_config_without_parse_yaml("configs/settings.yaml")

def set_settings_config(data):
    return _set_config("configs/settings.yaml",data)

def get_face_quality_config():
    return _get_config("configs/face_quality.yaml")

def get_face_quality_assessment_config():
    return _get_config("configs/face_quality_assessment.yaml")

def set_face_quality_assessment_config(data):
    return _set_config("configs/face_quality_assessment.yaml",data)

def get_models_config():
    return _get_config("configs/models.yaml")
