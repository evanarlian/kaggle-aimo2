#!/usr/bin/env bash

# # kaggle
# CUDA_VISIBLE_DEVICES=0 uv run vllm serve TODO --port=8001 &
# CUDA_VISIBLE_DEVICES=1 uv run vllm serve TODO --port=8002 &
# CUDA_VISIBLE_DEVICES=2 uv run vllm serve TODO --port=8003 &
# CUDA_VISIBLE_DEVICES=3 uv run vllm serve TODO --port=8004 &

# local
CUDA_VISIBLE_DEVICES=0 uv run vllm serve casperhansen/deepseek-r1-distill-qwen-1.5b-awq --gpu-memory-utilization=0.3 --max-model-len=5000 --port=8001 &
CUDA_VISIBLE_DEVICES=0 uv run vllm serve casperhansen/deepseek-r1-distill-qwen-1.5b-awq --gpu-memory-utilization=0.3 --max-model-len=5000 --port=8002 &

sleep 5

nginx -p nginx -c conf/nginx.conf
