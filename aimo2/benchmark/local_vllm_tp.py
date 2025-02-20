import argparse
import time

from vllm import LLM, SamplingParams


def main(model: str, n: int, tp: int):
    # inference
    llm = LLM(model, tensor_parallel_size=tp)
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
    # report
    print(f"Model: {model}")
    print(f"Concurrent requests: {n}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"Tokens per second: {tok_per_sec:.2f}")


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
    main(args.model, args.concurrent, args.tp)
