import argparse
import asyncio
import csv
import random
import string
from pathlib import Path

from openai import AsyncOpenAI


async def calc_token_usage(client: AsyncOpenAI, model: str) -> int:
    total_tokens = 0
    try:
        while True:
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            prefix = "".join(random.choices(string.printable, k=100))
            prompt = f"ID: {prefix} Solve 1/{a} + 1/{b}. Give the answer in fraction."
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=random.random(),
                top_p=random.random(),
            )
            assert completion.usage is not None
            total_tokens += completion.usage.completion_tokens
    except asyncio.CancelledError:
        pass
    finally:
        return total_tokens


async def main(model: str, batch_sizes: list[int], timeout: float):
    # write csv header on new file
    csv_file = Path("batch_saturation_result.csv")
    if not csv_file.exists():
        with csv_file.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_size", "total_tokens", "elapsed", "tok_per_sec"])

    # IMPORTANT! do not use nginx port, but rather single vLLM server port
    # this is because we want to benchmark single GPU only
    client = AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="-")
    for bs in batch_sizes:
        print(f"===== BATCH SIZE: {bs} =====")
        tasks = [
            asyncio.create_task(calc_token_usage(client, model)) for _ in range(bs)
        ]
        await asyncio.sleep(timeout)
        for t in tasks:
            t.cancel()
        total_tokens = sum(await asyncio.gather(*tasks))
        tok_per_sec = total_tokens / timeout if timeout > 0 else 0
        print(f"Total tokens generated: {total_tokens}")
        print(f"Total time taken: {timeout:.2f} seconds")
        print(f"Tokens per second: {tok_per_sec:.2f}")
        # append results to csv
        with csv_file.open("a") as f:
            writer = csv.writer(f)
            writer.writerow([bs, total_tokens, timeout, tok_per_sec])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="batch sizes (can be entered one or more times)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
        help="timeout in secs for a single conversation (longer == more accurate)",
    )
    args = parser.parse_args()
    print(args)
    asyncio.run(main(args.model, args.batch_sizes, args.timeout))
