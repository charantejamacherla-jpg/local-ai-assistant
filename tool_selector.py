import re


def should_use_calculator(message: str) -> bool:
    """
    Determine whether the user's message contains
    a mathematical calculation that should use
    the calculator tool.
    """

    if not isinstance(message, str):
        return False

    text = message.lower().strip()

    if not text:
        return False

    math_patterns = [
        r"\d+\s*[\+\-\*\/\%]\s*\d+",
        r"\d+\s*(plus|minus|times|multiplied by|divided by)\s*\d+",
        r"\bcalculate\b",
        r"\bwhat is\b.*\d+.*[\+\-\*\/\%]",
    ]

    return any(
        re.search(pattern, text)
        for pattern in math_patterns
    )


def select_tool(message: str) -> str | None:
    """
    Select an appropriate tool for the user's message.

    Returns the tool name or None.
    """

    if should_use_calculator(message):
        return "calculator"

    return None