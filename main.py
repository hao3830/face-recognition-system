import utils
import routers

import logging
import uvicorn

from fastapi import FastAPI
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware


SKIP_TRACK_TIME = 5
TIME_TRACK = 5

config = utils.get_config()

log_formatter = logging.Formatter("%(asctime)s %(levelname)s" " %(funcName)s(%(lineno)d) %(message)s")
log_handler = utils.BiggerRotatingFileHandler(
    "ali",
    config.log.dir,
    mode="a",
    maxBytes=2 * 1024 * 1024,
    backupCount=200,
    encoding=None,
    delay=0,
)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

logger.info("INIT LOGGER SUCCESSED")

#Init app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.middleware.origins,
    allow_credentials=config.middleware.allow_credentials,
    allow_methods=config.middleware.allow_methods,
    allow_headers=config.middleware.allow_headers,
)

#Router define
app.include_router(routers.camera)
app.include_router(routers.settings)

@app.get('/')
def healthy_check():
    return "OK"

if __name__ == '__main__':
    uvicorn.run(app,port=8000,host='0.0.0.0')


