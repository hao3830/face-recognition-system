import utils

import requests

from logging import getLogger

logger = getLogger("app")

config = utils.get_api_config()
checkin_config = config.checkin


def get_token():
    login_url = checkin_config.url + "/auth/login"
    try:
        res = requests.post(
            login_url,
            json={
                "username": checkin_config.username,
                "password": checkin_config.password,
            },
        )

        res = res.json()

        return res["tokens"]["access_token"]
    except Exception as err:
        logger.error(f"cannot sent get_token: {err}")


def good_bad_face(cropped_image):
    try:
        res = requests.post(
            url=config.good_bad_face_url, files=dict(binary_file=cropped_image)
        )
        res = res.json()

        if "predicts" in res:
            return res["predicts"][1]

        return None
    except Exception as err:
        logger.error(f"cannot sent good_bad_face: {err}")


def search_face(cropped_bytes, headers):
    check_url = checkin_config.url + "/person/check_face"
    try:
        res = requests.post(
            url=check_url, files=dict(file=cropped_bytes), headers=headers
        )
        return res
    except Exception as err:
        logger.error(f"cannot sent search_face: {err}")
        return None

def search_multiple_face(list_cropped_bytes, headers):
    check_url = checkin_config.url + "/person/check_multiple_face"
    list_cropped_bytes = [("files",cropped_bytes) for cropped_bytes in list_cropped_bytes]
    try:
        res = requests.post(
            url=check_url, files=list_cropped_bytes, headers=headers
        )
        return res
    except Exception as err:
        logger.error(f"cannot sent search_face: {err}")
        return None

def insert_face(cropped_bytes, headers):
    insert_url = checkin_config.url + "/person/search_face"
    try:
        res = requests.post(
            url=insert_url, files=dict(file=cropped_bytes), headers=headers
        )
        return res
    except Exception as err:
        logger.error(f"cannot sent insert_face: {err}")
        return None

def insert_multiple_face(list_cropped_bytes, headers):
    insert_url = checkin_config.url + "/person/search_multiple_face"
    list_cropped_bytes = [("files",cropped_bytes) for cropped_bytes in list_cropped_bytes]
    try:
        res = requests.post(
            url=insert_url, files=list_cropped_bytes, headers=headers
        )
        return res
    except Exception as err:
        logger.error(f"cannot sent insert_face: {err}")
        return None