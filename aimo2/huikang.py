import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def extract_boxed_text(text: str):
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def batch_message_filter(list_of_messages: list) -> tuple[list[list[dict]], list[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]["content"])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers


def select_answer(answers: list[str]):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
    return answer % 1000


def batch_message_generate(
    list_of_messages: list,
    max_model_len: int,
    cutoff_times: list[int],
    llm: LLM,
    tokenizer,
) -> list[list[dict]]:
    max_tokens = max_model_len
    if time.time() > cutoff_times[-1]:
        print("Speedrun")
        max_tokens = 2 * max_model_len // 3

    sampling_params = SamplingParams(
        temperature=1.0,  # Randomness of the sampling
        top_p=0.90,  # Cumulative probability of the top tokens to consider
        min_p=0.05,  # Minimum probability for a token to be considered
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=max_tokens,  # Maximum number of tokens to generate
        stop=["</think>"],  # List of strings that stop the generation
        seed=777,
    )

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )
        for messages in list_of_messages
    ]

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    print(
        [
            len(single_request_output.outputs[0].token_ids)
            for single_request_output in request_output
        ]
    )

    sort_keys_and_list_of_messages = []
    for messages, single_request_output in zip(list_of_messages, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print()
        messages.append(
            {"role": "assistant", "content": single_request_output.outputs[0].text}
        )

        sort_keys_and_list_of_messages.append(
            (len(single_request_output.outputs[0].token_ids), messages)
        )
    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    sort_keys_and_list_of_messages.sort(
        key=lambda sort_key_and_messages: sort_key_and_messages[0]
    )
    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])

    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]
    return list_of_messages


def create_starter_messages(question: str, index: int):
    options = []
    for _ in range(13):
        options.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000.",
                },
                {"role": "user", "content": question},
            ]
        )
    for _ in range(3):
        options.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
                },
                {"role": "user", "content": question},
            ],
        )
    return options[index % len(options)]


def predict_for_question(
    question: str,
    start_time: float,
    cutoff_time: float,
    cutoff_times: list[int],
    max_num_seqs: int,
    max_model_len: int,
    llm: LLM,
    tokenizer,
) -> int:
    # selected_questions_only = True
    selected_questions_only = False
    if selected_questions_only and not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        # if "Triangle" not in question:
        #    return 210
        if (
            "Triangle" not in question
            and "delightful" not in question
            and "George" not in question
        ):
            return 210

    if time.time() > cutoff_time:
        return 210

    print("🧊", question)

    num_seqs = max_num_seqs
    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * max_num_seqs // 3

    list_of_messages = [
        create_starter_messages(question, index) for index in range(num_seqs)
    ]

    all_extracted_answers = []
    for _ in range(1):
        list_of_messages = batch_message_generate(
            list_of_messages, max_model_len, cutoff_times, llm, tokenizer
        )

        if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            df = pd.DataFrame(
                {
                    "question": [question] * len(list_of_messages),
                    "message": [
                        messages[-1]["content"] for messages in list_of_messages
                    ],
                }
            )
            df.to_csv(f"{str(int(time.time() - start_time)).zfill(5)}.csv", index=False)

        list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
        all_extracted_answers.extend(extracted_answers)

    print("all_extracted_answers", all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    print("answer", answer)

    print("\n\n")
    cutoff_times.pop()
    return answer


def main():
    seed_everything(seed=0)
    start_time = time.time()
    cutoff_time = start_time + (4 * 60 + 45) * 60
    cutoff_times = [
        int(x) for x in np.linspace(cutoff_time, start_time + 60 * 60, 50 + 1)
    ]

    MAX_NUM_SEQS = 32
    MAX_MODEL_LEN = 8192 * 3 // 2

    llm_model_pth = "casperhansen/deepseek-r1-distill-qwen-1.5b-awq"

    llm = LLM(
        llm_model_pth,
        #    dtype="half",                 # The data type for the model weights and activations
        max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
        max_model_len=MAX_MODEL_LEN,  # Model context length
        trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
        tensor_parallel_size=1,  # The number of GPUs to use for distributed execution with tensor parallelism
        gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
        seed=2024,
    )
    tokenizer = llm.get_tokenizer()

    Q = "Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?"
    Q = "calculate sin(45) + cos(45)"

    predict_for_question(
        Q,
        start_time,
        cutoff_time,
        cutoff_times,
        MAX_NUM_SEQS,
        MAX_MODEL_LEN,
        llm,
        tokenizer,
    )


if __name__ == "__main__":
    main()
