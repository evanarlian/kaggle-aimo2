# https://github.com/casper-hansen/AutoAWQ/blob/main/examples/quantize.py
from argparse import ArgumentParser

from awq import AutoAWQForCausalLM
from huggingface_hub import create_repo, upload_folder
from transformers import AutoTokenizer


def main(model_path: str, revision: str):
    # 1. quantize
    quant_name = model_path.rsplit("/", maxsplit=1)[-1] + "-AWQ"
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    model = AutoAWQForCausalLM.from_pretrained(model_path, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.quantize(tokenizer, quant_config=quant_config)
    # 2. save to local filesystem
    model.save_quantized(quant_name)
    tokenizer.save_pretrained(quant_name)
    print(f"Quantized: {quant_name}")
    # 3. push to hub
    repo_id = f"evanarlian/{quant_name}"
    create_repo(repo_id, private=False, repo_type="model", exist_ok=True)
    upload_folder(folder_path=quant_name, repo_id=repo_id)
    print(f"Pushed: {repo_id}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="hf model")
    parser.add_argument("--revision", type=str, required=True, help="commit hash")
    args = parser.parse_args()
    print(args)
    main(args.model_path, args.revision)
