import random

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class FakeInput(BaseModel):
    model: str
    input: str


@app.post("/pooling")
async def fake_prm(req: FakeInput):
    n = req.input.count("<extra_0>")
    scores = []
    for _ in range(n):
        a = random.random()
        scores.append([1 - a, a])
    return {
        "id": "pool-fake-b186a9ae202a4ac5be022075c2b0d4c8",
        "object": "list",
        "created": 1743512565,
        "model": req.model,
        "data": [
            {
                "index": 0,
                "object": "pooling",
                "data": scores,
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "total_tokens": 42,
            "completion_tokens": 0,
            "prompt_tokens_details": None,
        },
    }


if __name__ == "__main__":
    print("🚀 Running fake PRM server for Qwen/Qwen2.5-Math-PRM-7B")
