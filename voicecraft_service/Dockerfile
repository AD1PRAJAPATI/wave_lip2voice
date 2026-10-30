FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
# VoiceCraft's config reads $USER, which does not exist in a container
ENV USER=voicecraft

RUN apt-get update && apt-get install -y \
    software-properties-common wget curl git build-essential ffmpeg espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# VoiceCraft is built for Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/pip/3.9/get-pip.py && python3.9 get-pip.py && rm get-pip.py
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && ln -sf /usr/bin/python3.9 /usr/bin/python3

WORKDIR /app
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

RUN git clone https://github.com/jasonppy/VoiceCraft.git /app/voicecraft
WORKDIR /app/voicecraft

# Must be an editable install (-e), otherwise audiocraft's config files are left out
RUN pip install --no-cache-dir -e \
    "git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft"
RUN pip install --no-cache-dir xformers==0.0.22

# Versions are pinned because newer ones conflict with each other
RUN pip install --no-cache-dir \
    phonemizer==3.2.1 num2words==0.5.13 huggingface_hub==0.22.2 torchmetrics==0.11.1 \
    transformers==4.38.2 tokenizers==0.15.2 tensorboard==2.16.2 openai-whisper \
    fastapi==0.110.0 "uvicorn[standard]==0.28.0" python-multipart==0.0.9

ENV VOICECRAFT_MODEL=330M_TTSEnhanced WHISPER_MODEL=base.en MODELS_DIR=/app/voicecraft/pretrained_models

# Models are downloaded at build time so the container starts fast
COPY download_models.py server.py ./
RUN python download_models.py

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
