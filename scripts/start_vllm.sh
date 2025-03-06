#!/usr/bin/env bash

CLONES=1

for i in $(seq 0 $((CLONES - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run vllm serve \
        casperhansen/deepseek-r1-distill-qwen-1.5b-awq \
        --port=$((8001 + i)) &
done

nginx -p nginx -c conf/nginx.conf

# TODO tune vllm settings, e.g. 0 logging
