from pathlib import Path

import pytest

from aimo2.parser.latex import MyLatexParser


@pytest.fixture
def parser():
    return MyLatexParser(grammar_dir=Path("aimo2/parser"), mod=1000, timeout=1.0)


def test_numbers(parser: MyLatexParser):
    assert parser.latex_to_int_modded("42") == 42
    assert parser.latex_to_int_modded("3.14") is None
    assert parser.latex_to_int_modded("-5") == 995
    assert parser.latex_to_int_modded("0") == 0


def test_basic_arithmetic(parser: MyLatexParser):
    assert parser.latex_to_int_modded("2 + 3") == 5
    assert parser.latex_to_int_modded("10 - 2") == 8
    assert parser.latex_to_int_modded("3 \\times 4") == 12
    assert parser.latex_to_int_modded("6 \\div 2") == 3


def test_fractions(parser: MyLatexParser):
    assert parser.latex_to_int_modded("\\frac{3}{2}") is None  # not whole int
    assert parser.latex_to_int_modded("\\frac{1}{4}") is None  # not whole int
    assert parser.latex_to_int_modded("\\frac{10}{5}") == 2  # is whole int


def test_simple_expressions(parser: MyLatexParser):
    assert parser.latex_to_int_modded("\\sqrt{16}") == 4
    assert parser.latex_to_int_modded("2^3") == 8
    assert parser.latex_to_int_modded("\\frac{3}{2} + 1") is None
    # NOTE: huggingface's math-verify-0.7.0 fails on below latex
    assert (
        parser.latex_to_int_modded("\\left\\lfloor 100 \\sqrt{2} \\right\\rfloor")
        == 141
    )


def test_failures(parser: MyLatexParser):
    assert parser.latex_to_int_modded("x + 1") is None  # Symbolic
    assert parser.latex_to_int_modded("\\frac{1}{0}") is None  # Division by zero
    assert parser.latex_to_int_modded("\\sqrt{-1}") is None  # Complex
    assert parser.latex_to_int_modded("\\text{hi}") is None  # Non-math
    assert parser.latex_to_int_modded("") is None  # Empty
