FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN pip3 install supervisely
RUN pip3 install "git+https://github.com/facebookresearch/detectron2.git"
RUN pip3 install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"