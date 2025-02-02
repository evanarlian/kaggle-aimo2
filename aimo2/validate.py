import asyncio
from argparse import ArgumentParser

from datasets import load_dataset
from openai import AsyncOpenAI


async def main():
    ds = load_dataset("AI-MO/aimo-validation-aime")
    ds = load_dataset("AI-MO/aimo-validation-amc")
    ds = load_dataset("AI-MO/aimo-validation-math-level-4")
    ds = load_dataset("AI-MO/aimo-validation-math-level-5")
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO use separate experiment jsons?
    # parser.add_argument("--model", type=str, required=True, help="vllm model")
    # parser.add_argument("--notes", type=str, required=True, help="extra notes (logged)")
    # parser.add_argument(
    #     "--repeats", type=int, required=True, help="num experiment iteration"
    # )

    args = parser.parse_args()
    asyncio.run(main())
