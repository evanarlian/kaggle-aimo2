import time

import requests


def wait_for_vllm(ports: list[int]) -> None:
    for port in ports:
        while True:
            print(".", end="", flush=True)
            try:
                r = requests.get(f"http://localhost:{port}/health")
                if r.status_code == 200:
                    print(f"vllm port {port} is ready")
                    break
            except Exception:
                pass
            time.sleep(1)
