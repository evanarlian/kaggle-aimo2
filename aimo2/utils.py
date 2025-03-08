import time
from datetime import datetime

import IPython.display as ipd
import pytz
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


def wib_now() -> str:
    """Generate current WIB time. Quite easy to read imho.
    Example: 2024-11-23__17.30.02
    """
    wib = pytz.timezone("Asia/Jakarta")
    timestamp = datetime.now(wib).strftime("%Y-%m-%d__%H.%M.%S")
    return timestamp
