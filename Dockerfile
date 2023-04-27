FROM ubuntu
WORKDIR /WORKDIR

RUN apt-get update && apt-get install python3-pip git -y
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx

RUN pip3 install -U pip

RUN pip install depthai opencv-python fastapi requests
RUN pip install PyYAML easydict
RUN pip install Pillow
RUN pip install blobconverter
RUN pip install mediapipe


CMD [ "bash", "start.sh" ]