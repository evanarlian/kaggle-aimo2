# kaggle-aimo2
AI Mathematical Olympiad - Progress Prize 2


# usage
Install deps
```bash
uv sync
```

Download competition dataset, this will also update the `kaggle_evaluation` dir with new code
```bash
./scripts/download_data.sh
```

Run majority voting based solution
```bash
./scripts/start_vllm_maj.sh

uv run -m aimo2.maj
```

Run reward model based solution
```bash
./scripts/start_vllm_bon.sh

uv run -m aimo2.bon
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


# notes
* this competition is all about inference time scaling:
  * majority-voting / self concistency (SC): easiest to do
  * best of N (BoN) (select the best answer according to reward/critic model): require good reward model, consumes 1 gpu slot
  * beam search
  * diverse tree verifier search (DVTS)
* awq is much faster than unsloth dynamic quants (bnb). On my machine, r1 1.5b, bs 16: awq (1423 tok/s) while bnb (399 tok/s). Need further investigation.
* [math-verify](https://github.com/huggingface/Math-Verify) by huggingface for parsing math, but fails on this latex -> `\\left\\lfloor 100 \\sqrt{2} \\right\\rfloor`. Answer should be 141.
* found out that using english prompt is better than using both english and chinese prompt (22 vs 17 score respectively). I removed the chinese prompt
* make sure vllm's `--max-seq-len-to-capture` covers your token range. Helps a little bit since the model does not fallback to eager execution. TODO try on kaggle's gpu
* stop at </ think>, this is because the llm sometimes tried to make the answer presentable to the user, which can be long.
* use temp 1.0 (default) because it will speed up a lot, need to check if the speed up worth not using lower temp.
* deepseek r1 series offical recommendation: https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations
* Qwen math PRM is not useful for steps inside deepseek's think token, there are no meaningful differences between wrong score and correct score, everything will be scored high.
* PRM does not work well in my case, putting the steps after </ think> token to PRM
