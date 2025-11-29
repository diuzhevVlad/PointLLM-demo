# Point-LLM Demo

This repository contains modified code of original [PointLLM](https://runsenxu.com/projects/PointLLM) project. 
In particular it provides fixed and improved version of Gradio demo (+ Docker) as part of study project for MIPT course.

## Set up
Load Docker image:
```
docker load vladislavdiuzhev/pointllm-demo
```
or build it locally:
```
docker compose build
```

## Run demo
Just run:
```
docker compose up
```
The process of loading model can take a while (even after downloading to cache in offline mode it is quite slow...)

After the message "Model loaded!" appeared in the terminal you can connect to the app in browser [0.0.0.0:7810](0.0.0.0:7810).

## Instructions
<div style="text-align: center;">
    <img src="assets/example.jpg" alt="Dialogue_Teaser" width=75% >
</div>

Proposed demo realizes 2 scenarios:

1. 3D Question Answering
2. 3D Understanding (coordinate-wise description)

### Hierarchy
You can choose point clouds from **data/chosen** directory (try to start the chat with something like "Describe point cloud in details").

**weights** directory will contain huggingface cache after initial download of the model (the second run can be performed without internet connection).

### Tips
- When using cpu 1 response can be quite long (up to several minutes)
- In 3D view you can examine coordinates
- Docker image size is ~ 20Gb as well as loaded weights