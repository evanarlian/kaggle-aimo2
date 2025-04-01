#!/usr/bin/env bash
set -Eeuxo pipefail

# nohup env CUDA_VISIBLE_DEVICES=0 vllm serve /kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1 --port=8001 > vllm8001.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=1 vllm serve /kaggle/input/open-r1/transformers/openr1-qwen-7b-awq/1/openr1_awq --port=8002 > vllm8002.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=2 vllm serve /kaggle/input/deepscaler/transformers/deepscaler-1.5b-preview-awq/1/deepscaler_awq --port=8003 > vllm8003.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=3 vllm serve /kaggle/input/qwen2.5-math/transformers/qwen2.5-math-prm-7b/1/qwen_math_prm --port=8004 > vllm8004.log 2>&1 &


MODEL=casperhansen/deepseek-r1-distill-qwen-1.5b-awq

uv run uvicorn aimo2.fake_prm_server:app --port=8004 --log-level=error &
uv run vllm serve $MODEL --port=8001 
