# huikang code flow
Reference: https://www.kaggle.com/code/huikang/deepseek-r1-distill-qwen-7b-awq

## initial setup
* seed, not that important
* cutoff_time is 4.75 hours after the competition starts
* cutoff_times in the duration from competition ends (4.75 hour mark) until (1 hour mark). You should read from the back (reversed).
* cutoff_times uses np.linspace, that means each new question will have 270 extra seconds (4.5 mins)
* The timing scheme is really weird. My intuition:
  * the earlier questions will have a very long time bonus
  * 1st question will have 3600 secs of runtime
  * 2nd question will have (3600 + 270 - elapsed so far) secs of runtime
  * 3rd question will have (3600 + 540 - elapsed so far) secs of runtime
  * ... and so on
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
* `predict_for_question`
  * the most important func for inference is , everything else stems from it
  * selected_questions_only is a flag used to skip the inference, I set it to False to simulate real competition condition
  * if time.time() > cutoff_time is interesting, if current time if more than the global limit (4.75 hours), the predict function will instantly return 210. Idk where did this number come from.
  * then create MAX_NUM_SEQS=32 messages. Each messages is a system prompt plus the question.
* call `batch_message_generate`
  * if current time is more than the desired per-question-cutoff-time (not the global one!), reduce MAX_MODEL_LEN to 8192. This is called **"Speedrun"**
  * make vllm SamplingParams:
    * temp 1.0
    * set max tokens either 12288 or 8192 based on speedrun or not
    * stop at `</think>`, i think this is to skip model yapping when it knows the final answer already inside the thinking part
    * seeded as well
  * manually converts regular (role and content) dict to raw llm string and use vllm LLM::generate method instead of the regular chat. Idk why
  * batch generate using vllm, get the same number of outputs as the inputs (32)
  * 1st array on stdout: print the generated token lengths
  * reconstruct the regular llm message (role and content) to add the generated output as assistant's output
  * 2nd array on stdout: same as 1st array
  * 3rd array on stdout: same as 1st array but sorted asc by token length
* call `batch_message_filter`
  * separates extractible and non-extractible content inside \\boxed{}
* call `select_answer`
  * for every extractible content, convert to int and add to counter
  * every counts will be added a small number for tiebreaker
  * if the counter is empty (either by no answer or failed int parsing), return 210 magic number

# recap
the most interesting things from the solution:
* timing, earlier questions are given more time, but the later will be speedrunning due to less time.
* prompt, mod 1000 are being performed from the model itself
