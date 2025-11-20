FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt update -y && apt install python3-pip libx11-dev libgl1-mesa-glx -y
RUN pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install -e .
