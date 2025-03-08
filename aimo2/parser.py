from typing import Optional

from sympy.parsing.latex import parse_latex


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


def latex_to_number(text: str) -> Optional[int | float]:
    try:
        # lark backend due to antlr not working for some reason
        sympy_expr = parse_latex(text, backend="lark")
        result = sympy_expr.evalf()
        if result.is_real:
            if result.is_integer:
                return int(result)
            return float(result)
        return None
    except Exception:
        return None
