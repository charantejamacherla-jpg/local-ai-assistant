def format_memory(messages: list[dict]) -> str:
    if not messages:
        return ""

    lines = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            label = "User"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = role.capitalize()

        lines.append(
            f"{label}: {content}"
        )

    return "\n".join(lines)