from aimo2.parser import latex_to_number


def test_numbers():
    assert latex_to_number("42") == 42
    assert latex_to_number("3.14") == 3.14
    assert latex_to_number("-5") == -5


def test_basic_arithmetic():
    assert latex_to_number("2 + 3") == 5
    assert latex_to_number("10 - 2") == 8
    assert latex_to_number("3 \\times 4") == 12
    assert latex_to_number("6 \\div 2") == 3


def test_fractions():
    assert latex_to_number("\\frac{3}{2}") == 1.5
    assert latex_to_number("\\frac{1}{4}") == 0.25
    assert latex_to_number("\\frac{10}{5}") == 2


def test_simple_expressions():
    assert latex_to_number("\\sqrt{16}") == 4
    assert latex_to_number("2^3") == 8
    assert latex_to_number("\\frac{3}{2} + 1") == 2.5
    assert latex_to_number("\\left\\lfloor 100 \\sqrt{2} \\right\\rfloor") == 141


def test_failures():
    assert latex_to_number("x + 1") is None  # Symbolic
    assert latex_to_number("\\frac{1}{0}") is None  # Division by zero
    assert latex_to_number("\\sqrt{-1}") is None  # Complex
    assert latex_to_number("\\text{hi}") is None  # Non-math
    assert latex_to_number("") is None  # Empty
