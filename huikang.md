# huikang code flow

## initial setup
* seed, not that important
* cutoff_time is 4.75 hours after the competition starts
* cutoff_times in the duration from competition ends (4.75 hour mark) until (1 hour mark). You should read from the back (reversed). This is really weird. My intuition:
  * the earlier questions will have a very long time bonus
  * 1st question will have 3600 secs of runtime
  * 2nd question will have (3600 - elapsed so far) secs of runtime
  * This way, earlier questions will not trigger "Speedrun" (explained later). Fully maxed out at MAX_MODEL_LEN=12288 tokens or found EOS token
  * Note that this might explain extreme score fluctuations in LB. Note that LB questions will be served randomly

## llm stuffs
* llm selected will always be r1-distill-qwen-7b-awq-casperhansen. QWQ-preview is just the relic of the past
* some important vllm LLM config:
  * max_model_len **might** be non existent parameter. I think max model length must be set individually from SamplingParam
  * tensor_parallel_size=4 to utilize all kaggle L4 gpus. I set it to 1 for local testing
  * gpu_memory_utilization=0.95, 0.05 more than the default. This should not matter that much. See: https://github.com/vllm-project/vllm/issues/1298
  * seed, interesting
* tokenizer is the standard model tokenizer

## inference 