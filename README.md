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
* can deepseek R1 distill model do TIR? qwen can do both CoT and TIR, but i think deepseek's CoT is superior
* interesting observation: wrong answers have longer CoTs. https://x.com/AlexGDimakis/status/1885447830120362099. Replicate this and try to exploit this as well
* prompt must be crafted to be similar to the training condition
* validation: use aime, math500, amc for validation. numina blogpost has these dataset ready to use for validation, but qwen math might be trained on them so i dunno, need to check. Use wandb for storing result.
* try min-p sampling

# Quick TODOS
* make AWQ tutorial. IMPORTANT! 🔥
* just submit once on kaggle to get the feel. IMPORTANT! 🔥
* just sleep and return 0 to test the timing. IMPORTANT! 🔥

# make it fast
* NOOO, the inference gateway is **forcing us to solve each question one by one**?? we cant leverage batching lol. Maybe we can use some crazy tricks like reference mutation, so that we can execute in parallel? If we cant cheat this system, we need to do 1 shot (or small shot) to stay under time limit (6min per Q).
* OR we can skip hard questions but the question comes randomly. i thought checking for hard questions can be done using thinking length, but that means we have to do the thinking first (wastes time), classic chicken and egg.
* vllm server and client approach can be used to squeeze gpu utilization (this will be super useful for ToRA TIR loop since each batch requires different treatment based on TIR result).
* vllm caching can be exploited if we ended up using RAT, because the reasoning is the same.
* fast python executor, benchmark the library loading speed. sympy. Deepseek says we can leverage exec() for warm interpreter
* tune vllm for even faster response
* maybe tune the thinking not to be too verbose?
* make cutoff time during inference so that we dont waste 

# just a thought
* i think the winning trick is to combine deepseek's strong reasoning with qwen TIR (RAT method). Plus maj@k.
