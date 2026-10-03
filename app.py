import os
import sys
from typing import Optional, Dict, Any

import requests
import laya

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
LAYA_MODEL = os.getenv("LAYA_MODEL", "english")
LAYA_PRELOAD = os.getenv("LAYA_PRELOAD", "false").lower() == "true"

# Classification questions (exact schema from test_laya.py)
CLASSIFICATION_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this message?",
        "criteria": {
            "billing": "invoices, payments, refunds, subscription issues",
            "technical": "bugs, crashes, outages, system errors",
            "account": "password resets, account access, profile changes",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["low", "medium", "high"]
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does the user need human assistance right now?"
    }
}

# Thresholds for routing decisions
NEEDS_HUMAN_THRESHOLD = 0.6  # If needs_human probability > 0.6, escalate to human


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


def classify_message(router: laya.Router, user_input: str) -> Dict[str, Any]:
    """
    Classify a user message using Laya.
    
    Args:
        router: Initialized Laya Router instance
        user_input: User's message text
    
    Returns:
        Dict with parsed classification:
        {
            "department": str (billing/technical/account/other),
            "urgency_label": str (low/medium/high),
            "urgency_score": float (continuous score),
            "needs_human": float (probability 0.0-1.0),
            "raw_result": dict (full Laya result)
        }
    """
    result = router.predict(user_input, CLASSIFICATION_QUESTIONS)
    answers = result["answers"]
    
    # Parse department (choice)
    dept_answer = answers["department"]
    department = dept_answer["choice"]
    
    # Parse urgency (score) - use highest probability to find winning category
    urgency_answer = answers["urgency"]
    urgency_score = urgency_answer["score"]
    urgency_probabilities = urgency_answer["probabilities"]
    urgency_legend = urgency_answer["legend"]
    
    # Find the category with highest probability
    winning_key = max(urgency_probabilities, key=urgency_probabilities.get)
    urgency_label = urgency_legend[winning_key]
    
    # Parse needs_human (noul - yes/no)
    needs_human_answer = answers["needs_human"]
    needs_human_prob = needs_human_answer["noul"]
    
    return {
        "department": department,
        "urgency_label": urgency_label,
        "urgency_score": urgency_score,
        "needs_human": needs_human_prob,
        "raw_result": result
    }


def route_decision(classification: Dict[str, Any]) -> Dict[str, str]:
    """
    Determine the routing action based on classification.
    
    Args:
        classification: Dict from classify_message()
    
    Returns:
        Dict with:
        {
            "action": "chat" | "escalate_human",
            "reason": str (explanation)
        }
    """
    needs_human_prob = classification["needs_human"]
    department = classification["department"]
    
    # If user needs human assistance, escalate
    if needs_human_prob > NEEDS_HUMAN_THRESHOLD:
        return {
            "action": "escalate_human",
            "reason": f"User needs human assistance (confidence: {needs_human_prob:.2f}). Escalating to {department} team."
        }
    
    # Otherwise, use conversational assistant
    return {
        "action": "chat",
        "reason": "Routing to conversational assistant."
    }


def display_classification(classification: Dict[str, Any]) -> None:
    """Display classification results to the user."""
    print("\n[Classification Results]")
    print(f"  Department: {classification['department']}")
    print(f"  Urgency: {classification['urgency_label']} (score: {classification['urgency_score']:.2f})")
    print(f"  Needs Human: {classification['needs_human']:.2f}")


def display_routing(routing: Dict[str, str]) -> None:
    """Display routing decision to the user."""
    print(f"[Routing] {routing['reason']}")


def main():
    print("Local AI Assistant with Smart Routing")
    print("Type 'exit' to quit.")
    print("-" * 56)

    # Initialize Laya Router once at startup
    print("Loading Laya classifier...")
    router = laya.Router(preload=LAYA_PRELOAD, default=LAYA_MODEL)
    print("Ready!\n")

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

        # Classify the message
        classification = classify_message(router, user_input)
        display_classification(classification)

        # Route based on classification
        routing = route_decision(classification)
        display_routing(routing)

        # Execute the routing action
        if routing["action"] == "chat":
            reply = ask_model(user_input, system_prompt=system_prompt)
            print(f"\nAssistant: {reply}")
        elif routing["action"] == "escalate_human":
            print(f"\n[ESCALATED] {routing['reason']}")


if __name__ == "__main__":
    main()
