from aimo2.parser import extract_boxed_text


def test_simple_content():
    assert extract_boxed_text(r"The answer is \(\boxed{42}\)") == "42"
    assert extract_boxed_text(r"Result: \(\boxed{3.14}\)") == "3.14"
    assert extract_boxed_text(r"\(\boxed{-5}\)") == "-5"


def test_basic_latex_constructs():
    assert extract_boxed_text(r"\(\boxed{\frac{3}{2}}\)") == r"\frac{3}{2}"
    assert extract_boxed_text(r"\(\boxed{\sqrt{16}}\)") == r"\sqrt{16}"
    assert extract_boxed_text(r"\(\boxed{2 + 3}\)") == "2 + 3"
    assert extract_boxed_text(r"\(\boxed{2^3}\)") == "2^3"


def test_nested_braces():
    assert extract_boxed_text(r"\(\boxed{2 + {3 + 4}}\)") == "2 + {3 + 4}"
    assert (
        extract_boxed_text(r"\(\boxed{\sqrt{{2 + 2} \times 4}}\)")
        == r"\sqrt{{2 + 2} \times 4}"
    )
    assert extract_boxed_text(r"\(\boxed{a + {b + {c + d}}}\)") == "a + {b + {c + d}}"


def test_complex_expressions():
    assert (
        extract_boxed_text(r"\(\boxed{\frac{3}{2} + \sqrt{16}}\)")
        == r"\frac{3}{2} + \sqrt{16}"
    )
    assert extract_boxed_text(r"\(\boxed{2 \times (3 + 4)}\)") == r"2 \times (3 + 4)"
    assert extract_boxed_text(r"\(\boxed{x^2 + y}\)") == "x^2 + y"


def test_no_boxed_content():
    assert extract_boxed_text(r"The answer is 42") is None
    assert extract_boxed_text(r"\(\sqrt{16}\)") is None
    assert extract_boxed_text(r"Plain text") is None
    assert extract_boxed_text(r"") is None


def test_multiple_boxed():
    assert extract_boxed_text(r"\(\boxed{5}\) and \(\boxed{10}\)") == "5"
    assert extract_boxed_text(r"\(\boxed{a}\) then \(\boxed{b}\)") == "a"


def test_malformed_or_tricky():
    assert extract_boxed_text(r"\(\boxed{unclosed\)") is None  # No closing brace
    assert extract_boxed_text(r"\(\boxed{}\)") == ""  # Empty content
    assert extract_boxed_text(r"\(\boxed{\text{hello}}\)") == r"\text{hello}"
    assert extract_boxed_text(r"\(\boxed{\frac{1}{0}}\)") == r"\frac{1}{0}"


def test_edge_cases_with_text():
    assert extract_boxed_text(r"Before \(\boxed{42}\) after") == "42"
    assert extract_boxed_text(r"\(\boxed{a}\) \(\sqrt{b}\)") == "a"
    assert extract_boxed_text(r"Text with \\boxed{xyz} in it") == "xyz"
