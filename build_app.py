import os
import sys
from typing import Any, Dict, Optional

import requests
import laya


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

LAYA_MODEL = os.getenv("LAYA_MODEL", "english")
LAYA_PRELOAD = os.getenv("LAYA_PRELOAD", "false").lower() == "true"

NEEDS_HUMAN_THRESHOLD = 0.60


CLASSIFICATION_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this message?",
        "criteria": {
            "billing": "invoices, payments, refunds, subscription issues",
            "technical": "bugs, crashes, outages, system errors",
            "account": "password resets, account access, profile changes",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["low", "medium", "high"],
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does the user need human assistance right now?",
    },
}


router = laya.Router(
    preload=LAYA_PRELOAD,
    default=LAYA_MODEL,
)


def ask_model(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
) -> str:
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
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.RequestException as exc:
        print(f"Error calling local model: {exc}", file=sys.stderr)
        return "Could not connect to the local AI model. Make sure Ollama is running."


def classify_message(message: str) -> Dict[str, Any]:
    result = router.predict(
        message,
        CLASSIFICATION_QUESTIONS,
    )

    answers = result["answers"]

    department_answer = answers["department"]
    urgency_answer = answers["urgency"]
    needs_human_answer = answers["needs_human"]

    department = department_answer["choice"]

    urgency_score = urgency_answer["score"]

    urgency_probabilities = urgency_answer.get("probabilities", {})

    if urgency_probabilities:
        urgency_key = max(
            urgency_probabilities,
            key=urgency_probabilities.get,
        )

        urgency_label = urgency_answer["legend"][str(urgency_key)]
    else:
        urgency_label = "unknown"

    needs_human = needs_human_answer["noul"]

    return {
        "department": department,
        "urgency": urgency_label,
        "urgency_score": urgency_score,
        "needs_human": needs_human,
        "raw": result,
    }


def route_decision(classification: Dict[str, Any]) -> Dict[str, str]:
    """
    Determine the routing action based on classification.
    """

    needs_human_prob = classification["needs_human"]
    department = classification["department"]

    if needs_human_prob > NEEDS_HUMAN_THRESHOLD:
        return {
            "action": "escalate_human",
            "reason": (
                f"User needs human assistance "
                f"(confidence: {needs_human_prob:.2f}). "
                f"Escalating to {department} team."
            ),
        }

    return {
        "action": "chat",
        "reason": "Routing to conversational assistant.",
    }


def main():
    print("Local AI Assistant with Smart Routing")
    print("Type 'exit' to quit.")
    print("-" * 56)

    system_prompt = (
        "You are a helpful local assistant. "
        "Be concise, friendly, and practical. "
        "Do not claim to have internet access or cloud capabilities. "
        "Keep responses clear and useful."
    )

    print("Loading Laya classifier...")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            classification = classify_message(user_input)

            decision = route_decision(classification)

            print(
                f"\nDepartment: "
                f"{classification['department']}"
            )

            print(
                f"Urgency: "
                f"{classification['urgency']} "
                f"(score: {classification['urgency_score']:.2f})"
            )

            print(
                f"Needs human: "
                f"{classification['needs_human']:.2f}"
            )

            print(
                f"Route: "
                f"{decision['action']}"
            )

            print(
                f"Reason: "
                f"{decision['reason']}"
            )

            if decision["action"] == "escalate_human":
                print(
                    "\nAssistant: "
                    "Your request has been identified as "
                    "needing human assistance."
                )

                print(
                    f"It has been routed to the "
                    f"{classification['department']} team."
                )

            else:
                reply = ask_model(
                    user_input,
                    system_prompt=system_prompt,
                )

                print(f"\nAssistant: {reply}")

        except Exception as exc:
            print(
                f"\nError processing request: {exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()