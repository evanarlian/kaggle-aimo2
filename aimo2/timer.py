import time


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
