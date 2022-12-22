FROM supervisely/base-py-sdk:6.68.60

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y ffmpeg libsm6 libxext6  
RUN apt-get install ninja-build

RUN pip3 install torch==1.13.0 torchvision==0.14.0
RUN pip3 install -U openmim
RUN mim install mmcv-full==1.7.0

RUN pip3 install mmsegmentation==0.29.1
RUN pip3 install mmcls==0.24.1
RUN pip3 install pyyaml==6.0
RUN pip3 install ffmpeg-python==0.2.0

RUN python3 -m pip install --upgrade supervisely