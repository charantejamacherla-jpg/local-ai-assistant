
from agent import run_agent


def test_calculator_action():
    result = run_agent("What is 127 * 84?")
    assert result == {
        "action": "calculator",
        "input": "127 * 84",
        "observation": "10668",
        "success": True,
    }
    print("PASS: structured calculator action")


def test_second_calculation():
    result = run_agent("Calculate 25 + 17")
    assert result is not None
    assert result["observation"] == "42"
    assert result["success"] is True
    print("PASS: second calculation")


def test_parentheses():
    result = run_agent("Calculate (100 / 4) + 7")
    assert result is not None
    assert result["observation"] == "32.0"
    assert result["success"] is True
    print("PASS: parentheses calculation")


def test_normal_question():
    assert run_agent("What are your business hours?") is None
    print("PASS: normal question rejection")


def test_empty_message():
    assert run_agent("") is None
    print("PASS: empty message rejection")


def test_invalid_input_type():
    assert run_agent(None) is None
    print("PASS: invalid input type protection")


def run_tests():
    print("=" * 50)
    print("PHASE 8D AGENT TEST SUITE")
    print("=" * 50)

    test_calculator_action()
    test_second_calculation()
    test_parentheses()
    test_normal_question()
    test_empty_message()
    test_invalid_input_type()

    print("=" * 50)
    print("ALL AGENT TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()