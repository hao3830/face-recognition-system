import utils

import requests

config = utils.get_api_config()
checkin_config = config.checkin

def get_token():
    login_url = checkin_config.url + "/auth/login"

    res = requests.post(login_url, json={
        "username": checkin_config.username,
        "password": checkin_config.password
    })
    
    res = res.json()
    
    return res["tokens"]["access_token"]

def good_bad_face(cropped_image):
    res = requests.post(
        url=config.good_bad_face_url,
        files=dict(binary_file=cropped_image)
    )

    res = res.json()
    
    
    if "predicts" in res:
        return res['predicts'][0]
    
    return None
