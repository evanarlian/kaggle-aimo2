import argparse
import asyncio
import random
import string
import time

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm


async def calc_completion_tokens(client: AsyncOpenAI, model: str) -> int:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Solve 1/2 + 1/3. Give the answer in fraction.",
            }
        ],
    )
    assert completion.usage is not None
    return completion.usage.completion_tokens


async def simple(client: AsyncOpenAI, model: str, n: int) -> None:
    tasks = []
    for _ in range(n):
        task = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Solve 1/2 + 1/3. Give the answer in fraction.",
                }
            ],
        )
        tasks.append(task)

    t0 = time.perf_counter()
    completions = await atqdm.gather(*tasks)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(c.usage.completion_tokens for c in completions)
    tok_per_sec = total_tokens / elapsed
    print("===== SIMPLE BENCHMARK =====")
    print(f"Concurrent requests: {n}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"Tokens per second: {tok_per_sec:.2f}")


async def randomized(client: AsyncOpenAI, model: str, n: int) -> None:
    # NOTE:
    # * we use random here to mimic real world, also to eliminate vllm same-sentence optimization (if any)
    # * setting temperature other than 1.0 will degrade perf, about 90% the original tok/s (RTX 3060)
    # * setting top_p other than 1.0 will degrade perf, about 66% the original tok/s (RTX 3060)
    tasks = []
    for _ in range(n):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        prefix = "".join(random.choices(string.printable, k=100))
        prompt = f"ID: {prefix} Solve 1/{a} + 1/{b}. Give the answer in fraction."
        task = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=random.random(),
            top_p=random.random(),
        )
        tasks.append(task)
    t0 = time.perf_counter()
    completions = await atqdm.gather(*tasks)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(c.usage.completion_tokens for c in completions)
    tok_per_sec = total_tokens / elapsed
    print("===== RANDOMIZED BENCHMARK =====")
    print(f"Concurrent requests: {n}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"Tokens per second: {tok_per_sec:.2f}")


async def main(model: str, n: int):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="-")
    await simple(client, model, n)
    await randomized(client, model, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument(
        "--concurrent", type=int, required=True, help="n concurrent requests"
    )
    args = parser.parse_args()
    print(args)
    asyncio.run(main(args.model, args.concurrent))
