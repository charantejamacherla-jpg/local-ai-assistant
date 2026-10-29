import os
import requests

from memory_manager import MemoryManager
from memory_formatter import format_memory


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b",
)


def ask_model(
    prompt: str,
    system_prompt: str,
) -> str:

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 100,
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "response",
        "",
    ).strip()


def main():

    memory = MemoryManager(
        max_turns=5
    )

    memory.remember_user(
        "My name is Alex."
    )

    memory.remember_assistant(
        "Nice to meet you, Alex."
    )

    conversation = format_memory(
        memory.get_context()
    )

    system_prompt = (
        "You are a helpful local assistant. "
        "Use the previous conversation when "
        "answering the current question. "
        "Do not invent information."
        "\n\n"
        "PREVIOUS CONVERSATION:\n"
        + conversation
    )

    question = "What is my name?"

    print(
        "Previous conversation:"
    )

    print(conversation)

    print(
        "\nCurrent question:"
    )

    print(question)

    reply = ask_model(
        question,
        system_prompt,
    )

    print(
        "\nAssistant:"
    )

    print(reply)


if __name__ == "__main__":
    main()