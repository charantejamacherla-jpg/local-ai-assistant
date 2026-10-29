from tools.calculator import calculate
from tools.registry import get_tool, list_tools


def test_basic_calculation():
    result = calculate("127 * 84")

    assert result == "10668"

    print("PASS: basic calculation")


def test_expression_precedence():
    result = calculate("25 + 17 * 2")

    assert result == "59"

    print("PASS: operator precedence")


def test_parentheses():
    result = calculate("(100 / 4) + 7")

    assert result == "32.0"

    print("PASS: parentheses")


def test_empty_expression():
    try:
        calculate("")
    except ValueError:
        print("PASS: empty expression protection")
        return

    raise AssertionError("Empty expression was accepted")


def test_invalid_characters():
    try:
        calculate("hello")
    except ValueError:
        print("PASS: invalid character protection")
        return

    raise AssertionError("Invalid characters were accepted")


def test_python_code_blocked():
    try:
        calculate("__import__('os')")
    except ValueError:
        print("PASS: Python code protection")
        return

    raise AssertionError("Python code was accepted")


def test_tool_registry():
    tool = get_tool("calculator")

    assert tool is not None
    assert tool["name"] == "calculator"
    assert callable(tool["function"])

    print("PASS: tool registry")


def test_tool_listing():
    tools = list_tools()

    assert len(tools) >= 1
    assert any(
        tool["name"] == "calculator"
        for tool in tools
    )

    print("PASS: tool listing")


def run_tests():
    print("=" * 50)
    print("TOOL TEST SUITE")
    print("=" * 50)

    test_basic_calculation()
    test_expression_precedence()
    test_parentheses()
    test_empty_expression()
    test_invalid_characters()
    test_python_code_blocked()
    test_tool_registry()
    test_tool_listing()

    print("=" * 50)
    print("ALL TOOL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()