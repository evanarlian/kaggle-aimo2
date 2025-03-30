from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import sympy
from lark import Lark
from sympy.parsing.latex.lark.transformer import TransformToSymPyExpr


# tutorial from: sympy/parsing/tests/test_custom_latex.py
class MyTransformToSymPy(TransformToSymPyExpr):
    def factorial2(self, tokens):
        return sympy.factorial2(tokens[0])

    def factorial(self, tokens):
        return sympy.factorial(tokens[0])


# modified from: sympy/parsing/latex/lark/latex_parser.py
class MyLatexParser:
    def __init__(self, grammar_dir: Path, mod: int, timeout: float):
        with open(grammar_dir / "latex.lark", "r") as f:
            latex_grammar = f.read()
        self.parser = Lark(
            latex_grammar,
            import_paths=[grammar_dir],
            parser="earley",
            start="latex_string",
            lexer="auto",
            ambiguity="explicit",
            propagate_positions=False,
            maybe_placeholders=False,
            keep_all_tokens=True,
        )
        self.transformer = MyTransformToSymPy()
        self.mod = mod
        self.timeout = timeout

    def doparse(self, s: str):
        # HACK: this is extremely dirty hack to handle double factorial (x!!), which is different from (x!)!
        # I can't find a clean way to parse !! using lark rules, it will parse into ambiguity nodes instead of just one (DOUBLE_BANG or BANG)
        # so the hack is to replace !! with @ and treat double factorial as entirely different token
        s = s.replace("!!", "@")
        parse_tree = self.parser.parse(s)
        sympy_expression = self.transformer.transform(parse_tree)
        return sympy_expression

    def _latex_to_int_modded(self, latex_str: str, queue: Queue):
        sympy_expr = self.doparse(latex_str)
        if not sympy_expr.is_integer:
            return
        modded_expr = sympy_expr % self.mod
        queue.put(int(modded_expr))

    def latex_to_int_modded(self, latex_str: str) -> Optional[int]:
        q = Queue(maxsize=1)  # oneshot queue
        p = Process(target=self._latex_to_int_modded, args=(latex_str, q))
        p.start()
        p.join(timeout=self.timeout)
        proc_result = q.get() if not q.empty() else None
        return proc_result


def extract_boxed_text(text: str) -> Optional[str]:
    """Extracts the first correct boxed content. If found, the content is returned as is (string)."""
    i = text.find("\\boxed{")
    if i == -1:
        return None
    text = text[i + 7 :]  # eat the '... \boxed{' string
    unmatched_left = 0
    captured = []
    for c in text:
        if c == "{":
            unmatched_left += 1
        elif c == "}":
            unmatched_left -= 1
        if unmatched_left < 0:  # found actual the closing brace
            return "".join(captured)
        captured.append(c)
    return None
