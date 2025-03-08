from aimo2.parser import latex_to_int


def test_numbers():
    assert latex_to_int("42") == 42
    assert latex_to_int("3.14") is None
    assert latex_to_int("-5") == -5


def test_basic_arithmetic():
    assert latex_to_int("2 + 3") == 5
    assert latex_to_int("10 - 2") == 8
    assert latex_to_int("3 \\times 4") == 12
    assert latex_to_int("6 \\div 2") == 3


def test_fractions():
    assert latex_to_int("\\frac{3}{2}") is None  # not whole int
    assert latex_to_int("\\frac{1}{4}") is None  # not whole int
    assert latex_to_int("\\frac{10}{5}") == 2  # is whole int


def test_simple_expressions():
    assert latex_to_int("\\sqrt{16}") == 4
    assert latex_to_int("2^3") == 8
    assert latex_to_int("\\frac{3}{2} + 1") is None
    # NOTE: huggingface's math-verify-0.7.0 fails on below latex
    assert latex_to_int("\\left\\lfloor 100 \\sqrt{2} \\right\\rfloor") == 141


def test_failures():
    assert latex_to_int("x + 1") is None  # Symbolic
    assert latex_to_int("\\frac{1}{0}") is None  # Division by zero
    assert latex_to_int("\\sqrt{-1}") is None  # Complex
    assert latex_to_int("\\text{hi}") is None  # Non-math
    assert latex_to_int("") is None  # Empty
