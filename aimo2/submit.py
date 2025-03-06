import asyncio
import os
import time
from collections import Counter
from typing import Optional

import polars as pl
from openai import AsyncOpenAI
from pydantic import BaseModel

import kaggle_evaluation.aimo_2_inference_server


class ConversationResult(BaseModel):
    pass


class Timer:
    def __init__(self, n_questions: int, time_limit: float):
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


async def conversation(q_text: str, client: AsyncOpenAI) -> ConversationResult:
    # TODO
    # implement repl, mod 1000, extraction etc
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
