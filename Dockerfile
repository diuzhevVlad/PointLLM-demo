FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG UID
ARG GID

RUN groupadd -g ${GID} usergroup && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash username

RUN apt update -y && apt install python3-pip libx11-dev libgl1-mesa-glx -y
RUN pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install -e .

USER username
