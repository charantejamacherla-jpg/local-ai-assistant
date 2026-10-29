from tool_selector import should_use_calculator, select_tool


def test_math_expression():
    assert should_use_calculator("What is 127 * 84?")
    print("PASS: math expression detection")


def test_word_math():
    assert should_use_calculator("Calculate 25 + 17")
    print("PASS: calculate keyword detection")


def test_normal_question():
    assert not should_use_calculator(
        "What are your business hours?"
    )
    print("PASS: normal question rejection")


def test_empty_message():
    assert not should_use_calculator("")
    print("PASS: empty message rejection")


def test_tool_selection():
    assert select_tool("What is 100 / 4?") == "calculator"
    assert select_tool("Tell me about your support") is None
    print("PASS: tool selection")


def run_tests():
    print("=" * 50)
    print("TOOL SELECTOR TEST SUITE")
    print("=" * 50)

    test_math_expression()
    test_word_math()
    test_normal_question()
    test_empty_message()
    test_tool_selection()

    print("=" * 50)
    print("ALL TOOL SELECTOR TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()