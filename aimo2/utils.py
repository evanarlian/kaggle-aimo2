import time

import IPython.display as ipd
import requests


def wait_for_vllm(ports: list[int], timeout: float) -> None:
    t0 = time.perf_counter()
    for port in ports:
        while True:
            print(".", end="", flush=True)
            try:
                r = requests.get(f"http://localhost:{port}/health")
                if r.status_code == 200:
                    print(f"vllm port {port} is ready")
                    break
            except Exception as e:
                elapsed = time.perf_counter() - t0
                if elapsed > timeout:
                    raise e
            time.sleep(1)


def mdlatex(text) -> None:
    """Modified deepseek completions tokens for easy viewing in jupyter notebook."""
    escaped = (
        text.replace("\\(", "$")
        .replace("\\)", "$")
        .replace("\\[", "$")
        .replace("\\]", "$")
        .replace("<think>", "***\\<think\\>***\n")
        .replace("</think>", "\n***\\</think\\>***")
    )
    ipd.display(ipd.Markdown(escaped))
