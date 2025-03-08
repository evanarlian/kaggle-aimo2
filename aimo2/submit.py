import asyncio
import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

import polars as pl
from openai import AsyncOpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer

import kaggle_evaluation.aimo_2_inference_server
from aimo2.parser import extract_boxed_text, latex_to_int
from aimo2.timer import Timer


class Prompt(BaseModel):
    language: Literal["en", "zh"]
    system: str
    boxed_enforcer: str
    # TODO add code prompt later


class ConversationResult(BaseModel):
    # TODO revamp later
    q_id: str
    boxed_answer: Optional[str]
    parsed_answer: Optional[int]
    elapsed: float
    language: Literal["en", "zh"]
    history: list[dict[Literal["role", "content"], str]]
    temperature: float
    top_p: float
    min_p: float


logging.basicConfig(level=logging.WARNING)  # sets 3rd party libs to WARNING
logger = logging.getLogger("aimo2")  # selects the whole `aimo2` module tree
logger.setLevel(logging.INFO)  # sets ours to INFO

# NOTE: change below according to the competition
N_PARALLEL = 32
MIN_VOTES = 15
MODEL = "casperhansen/deepseek-r1-distill-qwen-1.5b-awq"
timer = Timer(n_questions=10, time_limit=0.9 * 1 * 60 * 60)  # 90% of n hours
tokenizer = AutoTokenizer.from_pretrained(MODEL)
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="-")


def get_random_prompt() -> Prompt:
    en_system_prompts = [
        "Solve this math problem with a clear, step-by-step approach. Try a straightforward method and explain your reasoning. The answer is a whole integer, presented in \\boxed{}.",
        "Tackle this math problem using an alternative method from your usual approach. Show your steps briefly. The answer, a whole integer, goes in \\boxed{}.",
        "Analyze this math problem carefully, breaking it down logically. Focus on precision in your steps. The final whole integer answer must be in \\boxed{}.",
        "Explore this math problem by testing a key idea or shortcut. Explain your process simply. The answer is a whole integer, shown in \\boxed{}.",
        "Solve this math problem step-by-step, double-checking as you go. Keep it clear and concise. Place the whole integer answer in \\boxed{}.",
    ]
    zh_system_prompts = [
        "用清晰的步骤快速解决这个数学问题，解释你的推理。答案是整数，放在 \\boxed{} 中。",
        "用不同于常规的方法解决这个数学问题，简要展示步骤。答案是整数，写在 \\boxed{} 里。",
        "仔细分析这个数学问题，逻辑清晰地分解步骤。最终整数答案放在 \\boxed{} 内。",
        "通过尝试一个关键思路解决这个数学问题，简单说明过程。答案是整数，用 \\boxed{} 表示。",
        "一步步解决这个数学问题，边做边检查，保持简洁。整数答案写在 \\boxed{} 中。",
    ]
    en_boxed_enforcer = "\n\n**Final Answer:**\n\\[\n\\boxed{"
    zh_boxed_enforcer = "\n\n**答案是:**\n\\[\n\\boxed{"
    # TODO add code prompts too
    if random.random() < 0.5:
        prompt = Prompt(
            language="en",
            system=random.choice(en_system_prompts),
            boxed_enforcer=en_boxed_enforcer,
        )
    else:
        prompt = Prompt(
            language="zh",
            system=random.choice(zh_system_prompts),
            boxed_enforcer=zh_boxed_enforcer,
        )
    return prompt


async def conversation(q_text: str, q_id: str) -> ConversationResult:
    # 0. randomizer
    t0 = time.perf_counter()
    prompt = get_random_prompt()
    temperature = 0.7  # TODO make random later
    top_p = 0.95
    min_p = 0.05
    # 1. get answer initial answer
    history: Any = [
        {"role": "user", "content": q_text},
    ]
    completion1 = await client.chat.completions.create(
        model=MODEL,
        messages=history,
        temperature=temperature,
        top_p=top_p,
        extra_body={"min_p": min_p},
    )
    assert completion1.choices[0].message.content is not None
    reply1 = completion1.choices[0].message.content
    history.append({"role": "assistant", "content": reply1})
    # 2. force the model to output in the right format if not exist
    box_content1 = extract_boxed_text(reply1)
    logger.debug(f"[{q_id}] box_content1: {box_content1}")
    if box_content1 is None:
        # try to fix it once
        history[-1]["content"] = (
            history[-1]["content"].replace("</think>", "<think2>")
            + prompt.boxed_enforcer
        )
        raw = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        ).replace("<think2>", "</think>")  # type: ignore
        completion2 = await client.completions.create(
            model=MODEL,
            prompt=raw,
            temperature=temperature,
            top_p=top_p,
            extra_body={"min_p": min_p},
        )
        reply2 = history[-1]["content"] + completion2.choices[0].text
        history[-1]["content"] = reply2
        box_content2 = extract_boxed_text(reply2)
        logger.debug(f"[{q_id}] box_content2: {box_content2}")
        if box_content2 is None:
            # still no box after this, just bail
            return ConversationResult(
                q_id=q_id,
                boxed_answer=None,
                parsed_answer=None,
                elapsed=time.perf_counter() - t0,
                language=prompt.language,
                history=history,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
            )
        box_content1 = box_content2
    # 3. extract
    predicted_num = latex_to_int(box_content1)
    logger.debug(f"[{q_id}] predicted_num: {predicted_num}")
    if predicted_num is None:
        # unparsable number here
        return ConversationResult(
            q_id=q_id,
            boxed_answer=box_content1,
            parsed_answer=None,
            elapsed=time.perf_counter() - t0,
            language=prompt.language,
            history=history,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
        )
    # TODO implement repl and more, mod 1000, extraction etc
    # NOTE: the mod is being done at the very end so that we dont mess with history for the model
    logger.info(f"[{q_id}] convo done!")
    return ConversationResult(
        q_id=q_id,
        boxed_answer=box_content1,
        parsed_answer=predicted_num % 1000,
        elapsed=time.perf_counter() - t0,
        language=prompt.language,
        history=history,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
    )


async def worker(q_text: str, q_id: str, voting: Counter) -> list[ConversationResult]:
    convos = []
    try:
        while True:
            convo = await conversation(q_text, q_id)
            convos.append(convo)
            if convo.parsed_answer is not None:
                voting[convo.parsed_answer] += 1
    except Exception:
        logger.exception("[{q_id}] unexpected worker error")
    finally:
        return convos


async def monitor_voting(voting: Counter) -> None:
    assert MIN_VOTES > 0
    while True:
        await asyncio.sleep(0.1)
        total_votes = sum(voting.values())
        if total_votes < MIN_VOTES:
            continue
        answer, n_votes = voting.most_common(1)[0]
        if n_votes > total_votes // 2:
            break


async def solve_one(q_text: str, q_id: str) -> int:
    """Manages workers (parallel calls to vllm)"""
    allowed_time = timer.start_question()
    t0 = time.perf_counter()
    logger.info(f"[{q_id}] allowed time: {allowed_time}")
    voting = Counter()
    logger.info(f"[{q_id}] creating {N_PARALLEL} workers")
    worker_tasks = [
        asyncio.create_task(worker(q_text, q_id, voting)) for _ in range(N_PARALLEL)
    ]
    # waits for either timeout or monitor voting finishes first
    try:
        await asyncio.wait_for(monitor_voting(voting), timeout=allowed_time)
    except asyncio.TimeoutError:
        logger.info(f"[{q_id}] killing workers: timeout")
    else:
        elapsed = time.perf_counter() - t0
        logger.info(f"[{q_id}] killing workers: > 50% reached in {elapsed:0.2f} secs")
    # stop all workers and collect all the conversations
    for worker_task in worker_tasks:
        worker_task.cancel()
    convos_list = await asyncio.gather(*worker_tasks)
    # save to json
    all_convos = sum(convos_list, [])
    savepath = Path("all_convos.json")
    if savepath.exists():
        with open(savepath, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    existing += [c.model_dump() for c in all_convos]
    with open(savepath, "w") as f:
        json.dump(existing, f, indent=4)
    logger.info(f"[{q_id}] convos added to {savepath}")
    try:
        answer, n_votes = voting.most_common(1)[0]
    except IndexError:
        logger.warning(f"[{q_id}] there are no votes at all, convos might be too long")
        answer = 0
    # complete!
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
    prediction = asyncio.run(solve_one(q_text, id_))  # type: ignore
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
