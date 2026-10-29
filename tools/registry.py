from tools.calculator import calculate


TOOLS = {
    "calculator": {
        "name": "calculator",
        "description": (
            "Perform basic mathematical calculations."
        ),
        "function": calculate,
    }
}


def get_tool(name: str):
    return TOOLS.get(name)


def list_tools() -> list[dict]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
        }
        for tool in TOOLS.values()
    ]