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
* find the allowed models: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/548129.
* find vllm params, min-p sounds super good tbh!
* read last year winning solution. here is huikangs recap: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/546772. Note that from last year solution, there are path without training at all
* what is the difference between qwen instruct vs regular?
* can deepseek R1 distill model do TIR? qwen can do both CoT and TIR, but i think deepseek's CoT is superior

# Quick TODOS
* Need to get the AIME (or other math) score on above models. My sheet https://docs.google.com/spreadsheets/d/1iMsuLom4x3nkSus9htpK6r7vyeLTb7Odl5jftEej2oA/edit?gid=0#gid=0. IMPORTANT! 🔥. Complete the sheet
* Complete notion reading paper
* make AWQ tutorial

# just a thought
* i think the winning trick is to combine deepseek's strong reasoning with qwen TIR (RAT method). Plus maj@k.
