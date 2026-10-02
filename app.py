import os
import sys
from typing import Optional

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def ask_model(prompt: str, model: str = DEFAULT_MODEL, system_prompt: Optional[str] = None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 300,
        },
    }

    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as exc:
        print(f"Error calling local model: {exc}", file=sys.stderr)
        return "Could not connect to the local AI model. Make sure Ollama is running."


def main():
    print("Local AI Assistant Prototype")
    print("Type 'exit' to quit.")
    print("-" * 56)

    system_prompt = (
        "You are a helpful local assistant. Be concise, friendly, and practical. "
        "Do not claim to have internet access or cloud capabilities. "
        "Keep responses clear and useful."
    )

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        reply = ask_model(user_input, system_prompt=system_prompt)
        print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    main()
