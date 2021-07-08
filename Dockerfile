# FROM daisukekobayashi/darknet:gpu
FROM khanhmoriaty/yolov4_idcard_corner_detector:1.0

# COPY ./ ./
WORKDIR /

RUN apt-get update -y \ 
    && apt-get install -y python3 python3-pip git wget llvm libopencv-dev \
    && python3 -m pip install --upgrade pip
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip3 install opencv_python==4.2.0.34 uvicorn==0.14.0 fastapi==0.65.2 configparser==5.0.2