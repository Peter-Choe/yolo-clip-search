# CUDA 11.8 + cuDNN 기반 PyTorch 이미지
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 1. 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    wget curl git build-essential ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    unzip ca-certificates python3-pip python3-dev python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# 2. Miniconda 설치
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 3. Conda 환경 생성 및 faiss-gpu 포함 패키지 설치 (env name = image_search)
COPY environment.yaml /tmp/environment.yaml
RUN conda update -n base -c defaults conda && \
    conda env create -f /tmp/environment.yaml && \
    conda clean -a
ENV CONDA_DEFAULT_ENV=image_search
ENV PATH=$CONDA_DIR/envs/image_search/bin:$PATH

# 4. 작업 디렉토리
WORKDIR /app
COPY . /app

# 5. FastAPI 실행 포트
EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
