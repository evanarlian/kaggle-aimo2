import time

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
