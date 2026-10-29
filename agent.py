
from tool_selector import select_tool
from tool_input import extract_calculation
from tools.registry import get_tool


def prepare_tool_input(tool_name: str, user_input: str) -> str | None:
    """Prepare validated input for a selected tool."""
    if tool_name == "calculator":
        return extract_calculation(user_input)

    return None


def execute_tool(tool_name: str, tool_input: str) -> str:
    """Execute a registered tool with prepared input."""
    tool = get_tool(tool_name)

    if tool is None:
        raise ValueError("unknown tool")

    return tool["function"](tool_input)


def run_agent(user_input: str) -> dict | None:
    """Run one agent decision cycle and return a structured result."""
    if not isinstance(user_input, str):
        return None

    user_input = user_input.strip()

    if not user_input:
        return None

    selected_tool = select_tool(user_input)

    if not selected_tool:
        return None

    tool_input = prepare_tool_input(selected_tool, user_input)

    if tool_input is None:
        return {
            "action": selected_tool,
            "input": None,
            "observation": "Could not prepare valid tool input.",
            "success": False,
        }

    try:
        result = execute_tool(selected_tool, tool_input)

        return {
            "action": selected_tool,
            "input": tool_input,
            "observation": result,
            "success": True,
        }

    except (TypeError, ValueError):
        return {
            "action": selected_tool,
            "input": tool_input,
            "observation": "Tool execution failed safely.",
            "success": False,
        }