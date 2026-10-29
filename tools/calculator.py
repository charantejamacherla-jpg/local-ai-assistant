def calculate(expression: str) -> str:
    """
    Safely evaluate a basic mathematical expression.
    """

    allowed_characters = "0123456789+-*/().% "

    if not isinstance(expression, str):
        raise TypeError("expression must be a string")

    expression = expression.strip()

    if not expression:
        raise ValueError("expression cannot be empty")

    if any(
        character not in allowed_characters
        for character in expression
    ):
        raise ValueError("expression contains unsupported characters")

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as error:
        raise ValueError("invalid mathematical expression") from error

    return str(result)