import os
import random

import polars as pl

import kaggle_evaluation.aimo_2_inference_server


def mymodel(q_text: str) -> int:
    return random.randint(0, 999)


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    q_text = question.item(0)
    # Make a prediction
    prediction = mymodel(q_text)
    return pl.DataFrame({"id": id_, "answer": prediction})


def main():
    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(
        predict  # type: ignore
    )
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(("data/test.csv",))
        # sanity check
        df = pl.read_parquet("submission.parquet")
        print(df)


if __name__ == "__main__":
    main()
