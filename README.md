# kaggle-aimo2
AI Mathematical Olympiad - Progress Prize 2


# usage
Install deps
```bash
uv sync
```

Run majority voting
```bash
./scripts/start_vllm_maj.sh

uv run -m aimo2.maj
```

# development
Benchmark nginx token/sec. Edit the scripts to change settings.
```bash
./scripts/start_vllm.sh

uv run -m aimo2.benchmark.openai_nginx --model=casperhansen/deepseek-r1-distill-qwen-1.5b-awq --concurrent=100

killall nginx vllm
```
Benchmark local vllm (with tensor parallel) token/sec.
```bash
uv run -m aimo2.benchmark.local_vllm_tp --model=casperhansen/deepseek-r1-distill-qwen-1.5b-awq --concurrent=100 --tp=1
```

Benchmark at which batch size vLLM starts showing diminishing return of tok/sec. Note that this only measures single GPU (single vLLM server) for easier interpretation.
```bash
uv run -m aimo2.benchmark.batch_saturation --model=casperhansen/deepseek-r1-distill-qwen-1.5b-awq --batch-sizes 1 2 4 8 16 32 64 128 --timeout=60
```

Interesting findings about vLLM benchmark:
* Setting temperature other than 1.0 will degrade perf, about 90% the original tok/s (RTX 3060)
* Setting top_p other than 1.0 will degrade perf, about 66% the original tok/s (RTX 3060)

Quantize according to the whitelist rules.
```bash
uv run -m aimo2.quantize --model-path=agentica-org/DeepScaleR-1.5B-Preview --revision=24a92eff29154a702a928249812162644208ac5b

uv run -m aimo2.quantize --model-path=open-r1/OpenR1-Qwen-7B --revision=ae96ffba622dede862815c00d64270028a9ee8e4
```

Run tests
```bash
uv run pytest
```


# TODO
*  MODEL=evanarlian/DeepScaleR-1.5B-Preview-AWQ
* change vllm scripts to handle mutiple model later: prm, 3 llm
* test time scaling is not only majority voting (implemented!), there are: best of N (PRM based), beam search, dvts, etc
* increase n_parallel since we have used temp for speed demon
* do benchmark first just by counting the available json rows
* try qwen PRM model (non AWQ)
* try deepscaler1.5B (must be AWQ)
* try open R1 (must be AWQ)
* test-time scaling tricks:
  * majority voting (easiest)
  * best of N (select the best answer according to PRM)
  * beam search
  * DVTS
  * entropix
* reward model
  * standard PRM (Qwen PRM 7B is good)
  * entropy score (almost free to use)
* read last year winning solution. here is huikangs recap: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/546772. Note that from last year solution, there are path without training at all
  * 3rd (https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/517206)
    * parallel in batches
    * penalize small number <10, or numbers that appear on problem stmt
    * executing code during generation, the stuff codegen using logits processor. NOTE: how to do this using openai compatible server?
  * 4th (https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/518960)
    * use good evals
    * play with timings (TimeManager class)
    * dual gpu inference using threadpool
* interesting observation: wrong answers have longer CoTs. https://x.com/AlexGDimakis/status/1885447830120362099. Replicate this and try to exploit this as well
* validation: use aime, math500, amc for validation. numina blogpost has these dataset ready to use for validation, but qwen math might be trained on them so i dunno, need to check. Use wandb for storing result.


# vllm tune (do this on real kaggle submit kernel after modifying prompt forcing)
* preempt issues on check, see https://docs.vllm.ai/en/latest/performance/optimization.html.
* check vllm logs in kaggle to document mem used for gpu, for peak, for kv cache. Make sure the kv cache data type and use https://lmcache.ai/kv_cache_calculator.html to calculate.
* prompt forcing should add missing </ think> token if needed, check result first
* timer should allow one last time of giving answer, dont kill too early before prompt forcing
* maybe timer should be placed on worker instead


# vast ai todo
* dont do vast, not important for inference
* vast startup script change
* deactivate original venv (DONT nuke because it has jupyter)
* how to predownload using hf cli?
* setup nginx how
* disable workspace thingy and deactivate WORKSPACE line in bashrc
* try on cheap machines to learn about the vastai docker
* launch mode jupyter lab?


# notes
* awq is much faster than unsloth dynamic quants (bnb). On my machine, r1 1.5b, bs 16: awq (1423 tok/s) while bnb (399 tok/s). Need further investigation.
* [math-verify](https://github.com/huggingface/Math-Verify) by huggingface for parsing math, but fails on this latex -> `\\left\\lfloor 100 \\sqrt{2} \\right\\rfloor`. Answer should be 141.
* found out that using english prompt is better than using both english and chinese prompt (22 vs 17 score respectively). I removed the chinese prompt
* make sure vllm's `--max-seq-len-to-capture` covers your token range. Helps a little bit since the model does not fallback to eager execution. TODO try on kaggle's gpu
* stop at </ think>, this is because the llm sometimes tried to make the answer presentable to the user, which can be long
* use temp 1.0 (default) because it will speed up a lot
