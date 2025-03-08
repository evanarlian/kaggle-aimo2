import asyncio
import os
import random
import time
from collections import Counter
from typing import Literal, Optional

import polars as pl
from openai import AsyncOpenAI
from pydantic import BaseModel

import kaggle_evaluation.aimo_2_inference_server
from aimo2.parser import extract_boxed_text, latex_to_int


class Prompt(BaseModel):
    language: Literal["en", "zh"]
    system: str
    # TODO add code prompt later


class ConversationResult(BaseModel):
    boxed_answer: str
    elapsed: float
    language: Literal["en", "zh"]
    history: list[dict[Literal["system", "assistant", "user"], str]]
    temperature: float


# TODO move out
class Timer:
    def __init__(self, n_questions: int, time_limit: float):
        # TODO implement variable timing for hard tasks during early encounter
        # TODO look at other people timing code
        self.n_questions = n_questions
        self.time_limit = time_limit
        self.t0 = time.perf_counter()

    def start_question(self) -> float:
        remaining = self.time_limit - (time.perf_counter() - self.t0)
        return remaining / self.n_questions

    def finish_question(self) -> None:
        self.n_questions -= 1


# NOTE: change this according to the competition
timer = Timer(n_questions=10, time_limit=3600)


def get_random_prompt() -> Prompt:
    # TODO dont answer in whole integer!!, use smart parsing
    en_system_prompts = [
        "Solve this math problem with a clear, step-by-step approach. Try a straightforward method and explain your reasoning. The answer is a whole integer, presented in \\boxed{}.",
        "Tackle this math problem using an alternative method from your usual approach. Show your steps briefly. The answer, a whole integer, goes in \\boxed{}.",
        "Analyze this math problem carefully, breaking it down logically. Focus on precision in your steps. The final whole integer answer must be in \\boxed{}.",
        "Explore this math problem by testing a key idea or shortcut. Explain your process simply. The answer is a whole integer, shown in \\boxed{}.",
        "Solve this math problem step-by-step, double-checking as you go. Keep it clear and concise. Place the whole integer answer in \\boxed{}.",
    ]
    # TODO dont answer in whole integer!!, use smart parsing
    zh_system_prompts = [
        "用清晰的步骤快速解决这个数学问题，解释你的推理。答案是整数，放在 \\boxed{} 中。",
        "用不同于常规的方法解决这个数学问题，简要展示步骤。答案是整数，写在 \\boxed{} 里。",
        "仔细分析这个数学问题，逻辑清晰地分解步骤。最终整数答案放在 \\boxed{} 内。",
        "通过尝试一个关键思路解决这个数学问题，简单说明过程。答案是整数，用 \\boxed{} 表示。",
        "一步步解决这个数学问题，边做边检查，保持简洁。整数答案写在 \\boxed{} 中。",
    ]
    en_boxed_forcer = "So the final answer is \\boxed{"
    # TODO add code prompts too
    if random.random() < 0.5:
        prompt = Prompt(language="en", system=random.choice(en_system_prompts))
    else:
        prompt = Prompt(language="zh", system=random.choice(zh_system_prompts))
    return prompt


async def conversation(q_text: str, client: AsyncOpenAI) -> ConversationResult:
    # TODO
    # implement repl, mod 1000, extraction etc
    print(q_text)
    await asyncio.sleep(10)

    return ConversationResult()


async def worker(
    q_text: str, voting: Counter, client: AsyncOpenAI
) -> list[ConversationResult]:
    convos = []
    try:
        while True:
            convo = await conversation(q_text, client)
            convos.append(convo)
    except asyncio.CancelledError:
        return convos


async def monitor_voting(voting: Counter, min_votes: int) -> None:
    assert min_votes > 0
    while True:
        await asyncio.sleep(0.1)
        total_votes = sum(voting.values())
        if total_votes < min_votes:
            continue
        answer, n_votes = voting.most_common(1)[0]
        if n_votes > total_votes // 2:
            break


async def solve_one(q_text: str) -> int:
    """Manages workers (parallel calls to vllm)"""
    N_PARALLEL = 4
    MIN_VOTES = 11
    allowed_time = timer.start_question()
    print(allowed_time)
    # this client should point to nginx
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="-")
    voting = Counter()
    worker_tasks = [
        asyncio.create_task(worker(q_text, voting, client)) for _ in range(N_PARALLEL)
    ]
    # waits for either timeout or monitor voting finishes first
    try:
        await asyncio.wait_for(
            monitor_voting(voting, min_votes=MIN_VOTES), timeout=allowed_time
        )
    except asyncio.TimeoutError:
        print("Time's up!")
    else:
        print("Domination reached.")
    # stopping all workers, and collecting all the conversations
    for worker_task in worker_tasks:
        worker_task.cancel()
    convos_list = await asyncio.gather(*worker_tasks)
    all_convos = sum(convos_list, [])  # TODO save to file or smth
    try:
        answer, n_votes = voting.most_common(1)[0]
    except IndexError:
        print("There are no votes at all, convo might be too long.")
        answer = 0
    timer.finish_question()
    return answer


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(
    id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[int] = None
) -> pl.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    q_text = question.item(0)
    # Make a prediction
    prediction = asyncio.run(solve_one(q_text))
    return pl.DataFrame({"id": id_, "answer": prediction})


def main():
    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(
        predict  # type: ignore
    )
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(("data/test.csv",))
        # inference_server.run_local_gateway(("data/reference.csv",))
        # sanity check
        df = pl.read_parquet("submission.parquet")
        print(df)


if __name__ == "__main__":
    main()
