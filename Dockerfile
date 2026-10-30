FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common wget curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

RUN pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The model file (weights/pytorch_model.pt) is not copied into the image, it is mounted when the container runs
COPY app app
COPY lipreading lipreading
COPY espnet espnet
COPY assets assets

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
