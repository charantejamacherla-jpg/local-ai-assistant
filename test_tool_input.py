from tool_input import extract_calculation


def test_basic_expression():
    assert extract_calculation("What is 127 * 84?") == "127 * 84"
    print("PASS: basic expression extraction")


def test_addition():
    assert extract_calculation("Calculate 25 + 17") == "25 + 17"
    print("PASS: addition extraction")


def test_division():
    assert extract_calculation("What is 100 / 4?") == "100 / 4"
    print("PASS: division extraction")


def test_parentheses():
    assert extract_calculation("Calculate (100 / 4) + 7") == "(100 / 4) + 7"
    print("PASS: parentheses extraction")


def test_normal_question():
    assert extract_calculation("What are your business hours?") is None
    print("PASS: normal question rejection")


def test_empty_message():
    assert extract_calculation("") is None
    print("PASS: empty message rejection")


def run_tests():
    print("=" * 50)
    print("TOOL INPUT TEST SUITE")
    print("=" * 50)

    test_basic_expression()
    test_addition()
    test_division()
    test_parentheses()
    test_normal_question()
    test_empty_message()

    print("=" * 50)
    print("ALL TOOL INPUT TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()