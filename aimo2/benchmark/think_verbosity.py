import argparse
import asyncio
from typing import Optional

import pandas as pd
from openai import AsyncOpenAI


async def calc_tokens(
    client: AsyncOpenAI, model: str, system: Optional[str], question: str
) -> int:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stop=["</think>"],  # we only care about thinking tokens
        max_tokens=16000,  # to save time
    )
    assert completion.usage is not None
    return completion.usage.completion_tokens


async def main(model: str):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="-")
    questions = [
        "Calculate mean of [3, 4, 5]",
        "What is sin(90)?",
    ]
    systems = [
        None,
        "Be concise and put final answer in \\boxed{}",
        "Make step by step reasoning and put final answer in \\boxed{}",
    ]
    N = 32
    results = []
    for question in questions:
        for system in systems:
            print(f"benchmarking: question {question}, system {system}")
            tasks = [calc_tokens(client, model, system, question) for _ in range(N)]
            tokens = await asyncio.gather(*tasks)
            results.append({"question": question, "system": system, "tokens": tokens})
    df = pd.DataFrame(results)
    print(df)
    # for nice visualization see my notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name")
    args = parser.parse_args()
    print(args)
    asyncio.run(main(args.model))
