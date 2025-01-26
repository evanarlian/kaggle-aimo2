# kaggle-aimo2
AI Mathematical Olympiad - Progress Prize 2

# TODO
* Make the model reliably generate tool calls in its thinking phase. We can distill this by using open ended generation in vllm (with stop token). Quite complicated.
* Use chinese prompt as TTA. Can i use this chinese model as verifier model as well? We can do vice versa as well. https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/559418
* Output parsing. Mismatching output is exactly the issue huggingface team deal with. Somehow we need to parse latex to symbollic math before final answer. https://github.com/huggingface/Math-Verify
* We can use deepseek's reasoning to boost other model perf. https://x.com/skirano/status/1882819133043323359
* read numina math paper and list all the things that they did. nice video: https://www.youtube.com/watch?v=zNplyggkjbY&list=PLqFaTIg4myu9zFM9d23v8w5qKfaLvMu3i&index=1
* look at numina code and dataset
* look at huikang's TIR code
* look at vllm install code
* grab the inference server code from kaggle to run locally. IMPORTANT! 🔥
* find the allowed models: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/548129. Look at the allowed commits **IMPORTANT**!!
* find vllm params, min-p sounds super good tbh!
* Whitelisting: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/548129
* The rules are: commit must be released before 1 Oct 2024, license must be very permissive. Legends (❓= not yet whitelisted). I want to use (make sure to download just the commit before 1 Oct 2024). Better to put in spreadsheet and link here. IMPORTANT! 🔥:
  * Qwen/Qwen2.5-Math-72B-Instruct (qwen license❓, 23 Sep 2024)
  * Qwen/Qwen2.5-Math-7B-Instruct (apache license, 23 Sep 2024)
  * Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ (apache license, 20 Jan 2025❓)
  * Qwen/Qwen2.5-72B-Instruct (qwen license❓, 25 Sep 2024)
  * Qwen/Qwen2.5-32B-Instruct (apache license, 25 Sep 2024)
  * KirillR/QwQ-32B-Preview-AWQ (apache license, 28 Nov 2024)
  * deepseek-ai/DeepSeek-R1-Distill-Llama-70B (mit license, 23 Jan 2025)
  * what is PRM model in Qwen math, also there is another one i believe
* Need to get the AIME (or other math) score on above models.
* read last year winning solution. here is huikangs recap: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/546772. Note that from last year solution, there are path without training at all
