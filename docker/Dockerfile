FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# 保留你原来的系统依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends build-essential g++ gcc
RUN apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0
RUN apt-get install -y --no-install-recommends openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git

# ---- 安装 Miniconda（真正执行安装）----
ENV CONDA_DIR=/opt/conda
SHELL ["/bin/bash","-lc"]
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
 && rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

# ---- 接受 Anaconda 通道 ToS（解决你最初的 conda 报错）----
RUN conda config --set always_yes yes --set changeps1 no \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda clean -afy

# ---- 新建独立环境 bev，使用 Python 3.8 ----
RUN conda create -y -n bev python=3.8 && conda clean -afy
ENV CONDA_DEFAULT_ENV=bev
ENV PATH=/opt/conda/envs/bev/bin:/opt/conda/bin:${PATH}

# ---- 下面所有安装都进入 bev 环境（版本号与你原始保持一致）----
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate bev && \
    conda install -y -c pytorch pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 && \
    pip install Pillow==8.4.0 && \
    pip install tqdm && \
    pip install torchpack && \
    pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0 && \
    pip install nuscenes-devkit && \
    pip install mpi4py==3.0.3 && \
    pip install numba==0.48.0 && \
    pip install --no-cache-dir jupyterlab ipykernel

WORKDIR /workspace

