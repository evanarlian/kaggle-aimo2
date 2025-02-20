import argparse
import asyncio
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


async def main(model: str, n: int):
    # call many
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="-")
    tasks = [calc_completion_tokens(client, model) for _ in range(n)]
    t0 = time.perf_counter()
    tokens = await atqdm.gather(*tasks)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(tokens)
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
    args = parser.parse_args()
    asyncio.run(main(args.model, args.concurrent))
