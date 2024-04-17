FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
FROM tensorflow/tensorflow:2.9.0-gpu

# RUN groupadd -g 1000 docker && \
#     useradd --create-home -r -u 1000 -g docker docker && \
#     echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Force packages to be install non-interactively
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y xauth zsh git curl glmark2 lsb-release gnupg2 wget vim gcc g++ tmux swig libgl1 libglib2.0-0 libsm6 libxrender1 libxext6


# USER docker
# ENV PATH="$PATH:/home/docker/.local/bin"

RUN pip install opencv-python matplotlib debugpy


WORKDIR /workspace/cca