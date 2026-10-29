import re


def extract_calculation(message: str) -> str | None:
    """
    Extract a mathematical expression from a natural-language message.
    Returns the expression or None if no calculation is found.
    """
    if not isinstance(message, str):
        return None

    text = message.strip()

    if not text:
        return None

    # Find expressions such as:
    # 127 * 84
    # 25 + 17
    # 100 / 4
    # (100 / 4) + 7
    pattern = r"[\d\s()+\-*/%.]+"

    matches = re.findall(pattern, text)

    for match in matches:
        expression = match.strip()

        if not expression:
            continue

        # Must contain at least one digit and one math operator.
        if re.search(r"\d", expression) and re.search(r"[+\-*/%]", expression):
            return expression

    return None