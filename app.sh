#!/usr/bin/env sh

DEVICE_FLAGS=""
if [ "${START_CPU:-}" = "1" ] || [ "${START_CPU:-}" = "true" ]; then
    DEVICE_FLAGS="--device cpu --torch_dtype float32"
fi

APP_PORT="${APP_PORT:-7810}"

PYTHONPATH=$PWD python3 -W ignore pointllm/eval/chat_gradio_minimal.py $DEVICE_FLAGS --port "${APP_PORT}"
