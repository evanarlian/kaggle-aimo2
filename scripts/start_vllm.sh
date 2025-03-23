#!/usr/bin/env bash

CLONES=1
# TODO change to handle mutiple model later: prm, 3 llm
# MODEL=evanarlian/DeepScaleR-1.5B-Preview-AWQ
MODEL=casperhansen/deepseek-r1-distill-qwen-1.5b-awq

for i in $(seq 0 $((CLONES - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run vllm serve $MODEL --port=$((8001 + i)) &
done

nginx -p nginx -c conf/nginx.conf

# TODO tune vllm settings, e.g. 0 logging
