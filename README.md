# kaggle-aimo2
AI Mathematical Olympiad - Progress Prize 2

# usage
Install deps
```bash
uv sync
```

Benchmark token/sec
```bash
uv run -m aimo2.benchmark --model=casperhansen/deepseek-r1-distill-qwen-1.5b-awq --concurrent=100
```

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
* reward model can be used to prune. REBASE paper
* test time scaling with budget forcing paper. Literally just force add "Wait" in the sentence.
* Pay attention to entropy, how can i leverage entropy information
* try 70b reasoner?

# Quick TODOS
* make AWQ tutorial. IMPORTANT! 🔥
* just submit once on kaggle to get the feel. IMPORTANT! 🔥
* just sleep and return 0 to test the timing. IMPORTANT! 🔥
* try on real L4 gpu on vast
* run docker command vast locally to ensure im doing correct stuff (from the template from vast)

# make it fast
* NOOO, the inference gateway is **forcing us to solve each question one by one**?? we cant leverage batching lol. Maybe we can use some crazy tricks like reference mutation, so that we can execute in parallel? If we cant cheat this system, we need to do 1 shot (or small shot) to stay under time limit (6min per Q).
* OR we can skip hard questions but the question comes randomly. i thought checking for hard questions can be done using thinking length, but that means we have to do the thinking first (wastes time), classic chicken and egg.
* vllm server and client approach can be used to squeeze gpu utilization (this will be super useful for ToRA TIR loop since each batch requires different treatment based on TIR result).
* vllm caching can be exploited if we ended up using RAT, because the reasoning is the same.
* fast python executor, benchmark the library loading speed. sympy. Deepseek says we can leverage exec() for warm interpreter
* tune vllm for even faster response
* maybe tune the thinking not to be too verbose?
* make cutoff time during inference so that we dont waste
* running 4 vllms at the same time seem to be a good idea vs tensor parallel? check speed later

# just a thought
* i think the winning trick is to combine deepseek's strong reasoning with qwen TIR (RAT method). Plus maj@k.

# TIR hacking!!!!!!!!
* This is a much simpler way to vs RAT + TIR, whatever
* fyi qwen 32B distill is better at math vs qwen math, also better than coding vs qwen coder.
* we can force inject code inside the thinking (and the stdout outputs) and the model will continue the code
* beware of tabbed code like this, meed to be robust on the parser. The output should be tabbed as well to maintain coherence. Maybe use pydantic to store all parsed code.
    ```python
        a = 3
        print(a)
    ```
* idea: if error occurs, we can force the model to fix it by injecting "Okay I think I need to fix the code"
* idea: observe the model behavior during coding, sometimes the model is doing the code one by one, so we need to support stateful coding (like jupyter), maybe jupyter is the answer?
* the hardest part is how to tell the model that it can use tools like this, I think this is not trivial due to lack of TIR training data during distillation
* hmm can i combine some entropy magic (like entropix) to make a tree based decision?
* maybe not entropix, but the idea is, at first, there is 3 (or more!) concurrent calls to vllm, the first one is code directly injected, the second one is code injected but late, the 3rd one is full blown reasoning (the 3rd one can be mixed with running code AFTER reasoning finishes to make sure what it spits are correct). Each branch can split to 3 other branch. This is sooooooooooo damn natural because we can just keep branching until time's up, and at the end we just use majority voting. We might be able to solve easy questions as well!!!!! this is due to BFS-like approach, if the all the child returns the same value, then you know you have something.
* Managing time is quite challenging as well, can we adjust the time limit based on num questions left? e.g if we solve 49 easy, the last 1 we can take our sweet time, if time out then get the most vote (SC), or consensus reached
* the voting can be coded like this, set a counter, minimum vote is 7. if a number is more than 50%, just terminate all branch, and move on!
* we must nudge the model to use tools, it does not have to be perfect but think that tools are like calculator, we can whip it out like whenever we are stuck with a small part of the problem. how to prompt that?? Dont forget the prompt and 
* Dont forget to parse the boxed, do logic of mod OUTSIDE, this is to make the task easier for model



# vast ai todo
* vast startup script change
* deactivate original venv (DONT nuke because it has jupyter)
* how to predownload using hf cli?
* setup nginx how
* disable workspace thingy and deactivate WORKSPACE line in bashrc
* try on cheap machines to learn about the vastai docker
* launch mode jupyter lab?
* 