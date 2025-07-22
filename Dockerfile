# CUDA 11.8 + cuDNN 기반 PyTorch 이미지
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 1. 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    wget curl git build-essential ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    unzip ca-certificates python3-dev python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# 2. Miniconda 설치
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    bash ~/miniforge.sh -b -p $CONDA_DIR && \
    rm ~/miniforge.sh

# 3. Conda 환경 생성
COPY environment.yaml /tmp/environment.yaml
COPY requirements.txt /tmp/requirements.txt   
SHELL ["/bin/bash", "-c"]
RUN conda install -n base -c conda-forge mamba -y && \
    mamba env create -f /tmp/environment.yaml && \
    conda clean -a

# pip install 시 numpy 재설치 방지 + pydantic 고정
RUN conda run -n image_search pip install --no-deps -r /tmp/requirements.txt && \
    conda run -n image_search pip install numpy==1.26.4 --force-reinstall && \
    conda run -n image_search pip uninstall -y pydantic_core

# 5. 작업 디렉토리
WORKDIR /app
COPY . /app
RUN mkdir -p /app/models

# 6. 환경 변수 및 포트
ENV CONDA_DEFAULT_ENV=image_search
ENV PATH=$CONDA_DIR/envs/image_search/bin:$PATH
EXPOSE 8000
EXPOSE 8501
