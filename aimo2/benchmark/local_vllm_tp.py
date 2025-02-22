import argparse
import random
import string
import time

from vllm import LLM, SamplingParams


def simple(llm: LLM, n: int) -> None:
    messages = [
        [{"role": "user", "content": "Solve 1/2 + 1/3. Give the answer in fraction."}]
        for _ in range(n)
    ]
    sampling_params = SamplingParams(max_tokens=None)
    t0 = time.perf_counter()
    batch_completions = llm.chat(messages, sampling_params)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(c.outputs[0].token_ids) for c in batch_completions)
    tok_per_sec = total_tokens / elapsed
    print("===== SIMPLE BENCHMARK =====")
    print(f"Concurrent requests: {n}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"Tokens per second: {tok_per_sec:.2f}")


def randomized(llm: LLM, n: int) -> None:
    # NOTE:
    # * we use random here to mimic real world, also to eliminate vllm same-sentence optimization (if any)
    # * setting temperature other than 1.0 will degrade perf, about 90% the original tok/s (RTX 3060)
    # * setting top_p other than 1.0 will degrade perf, about 66% the original tok/s (RTX 3060)
    messages = []
    sampling_params = []
    for _ in range(n):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        prefix = "".join(random.choices(string.printable, k=100))
        prompt = f"ID: {prefix} Solve 1/{a} + 1/{b}. Give the answer in fraction."
        messages.append([{"role": "user", "content": prompt}])
        sampling_params.append(
            SamplingParams(
                max_tokens=None,
                temperature=random.random(),
                top_p=random.random(),
            )
        )
    t0 = time.perf_counter()
    batch_completions = llm.chat(messages, sampling_params)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(c.outputs[0].token_ids) for c in batch_completions)
    tok_per_sec = total_tokens / elapsed
    print("===== RANDOMIZED BENCHMARK =====")
    print(f"Concurrent requests: {n}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"Tokens per second: {tok_per_sec:.2f}")


def main(model: str, n: int, tp: int):
    llm = LLM(model, tensor_parallel_size=tp)
    simple(llm, n)
    randomized(llm, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument(
        "--concurrent", type=int, required=True, help="n concurrent requests"
    )
    parser.add_argument(
        "--tp", type=int, required=True, help="number of tensor parallel"
    )
    args = parser.parse_args()
    print(args)
    main(args.model, args.concurrent, args.tp)
