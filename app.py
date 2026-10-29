import os
import sys
from typing import Any, Dict, Optional

import requests
import laya

from vector_search import build_index, search
from memory_manager import MemoryManager
from memory_formatter import format_memory
from embeddings import get_embedding, cosine_similarity
from tool_selector import select_tool
from tool_input import extract_calculation
from tools.registry import get_tool


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

DEFAULT_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b",
)

LAYA_MODEL = os.getenv(
    "LAYA_MODEL",
    "english",
)

LAYA_PRELOAD = os.getenv(
    "LAYA_PRELOAD",
    "false",
).lower() == "true"

NEEDS_HUMAN_THRESHOLD = 0.60

RAG_MIN_SIMILARITY = 0.55

ANSWERABILITY_THRESHOLD = 0.72


CLASSIFICATION_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": (
            "Classify the user's request into the "
            "department that can best handle it."
        ),
        "criteria": {
            "billing": (
                "invoices, payments, refunds, "
                "subscription issues"
            ),
            "technical": (
                "bugs, crashes, outages, "
                "system errors"
            ),
            "account": (
                "password resets, account access, "
                "profile changes"
            ),
            "other": (
                "general information, business hours, "
                "holiday schedules, product information, "
                "sales information"
            ),
        },
    },

    "urgency": {
        "type": "score",
        "instructions": (
            "Estimate how urgent the user's request is."
        ),
        "criteria": [
            "low",
            "medium",
            "high",
        ],
    },

    "needs_human": {
        "type": "noul",
        "instructions": (
            "Determine whether the user needs assistance "
            "from a human support agent."
        ),
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

        return data.get(
            "response",
            "",
        ).strip()

    except requests.exceptions.RequestException as exc:

        print(
            f"Error calling local model: {exc}",
            file=sys.stderr,
        )

        return (
            "Could not connect to the local AI model. "
            "Make sure Ollama is running."
        )


def classify_message(
    message: str,
) -> Dict[str, Any]:

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

    urgency_probabilities = (
        urgency_answer.get(
            "probabilities",
            {},
        )
    )

    if urgency_probabilities:

        urgency_key = max(
            urgency_probabilities,
            key=urgency_probabilities.get,
        )

        urgency_label = (
            urgency_answer["legend"][
                str(urgency_key)
            ]
        )

    else:

        urgency_label = "unknown"

    needs_human = (
        needs_human_answer["noul"]
    )

    return {
        "department": department,
        "urgency": urgency_label,
        "urgency_score": urgency_score,
        "needs_human": needs_human,
        "raw": result,
    }


def apply_department_override(
    message: str,
    classification: Dict[str, Any],
) -> Dict[str, Any]:

    text = message.lower()

    general_information_terms = [
        "business hours",
        "opening hours",
        "working hours",
        "holiday schedule",
        "pricing plans",
        "price plans",
        "product information",
        "sales information",
    ]

    if any(
        term in text
        for term in general_information_terms
    ):

        classification["department"] = "other"

    return classification


def route_decision(
    classification: Dict[str, Any],
) -> Dict[str, str]:

    needs_human_prob = (
        classification["needs_human"]
    )

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
        "reason": (
            "Routing to conversational assistant."
        ),
    }


def retrieval_is_answerable(
    user_query: str,
    knowledge_item: Dict[str, Any],
) -> bool:
    """
    Determine whether the retrieved knowledge can answer
    the user's actual question.

    Uses semantic similarity plus deterministic intent
    protection for questions that ask for information
    the knowledge base does not contain.
    """

    answerable_for = knowledge_item.get(
        "answerable_for",
        [],
    )

    if not answerable_for:
        return False

    query = user_query.lower().strip()

    # -------------------------------------------------
    # Protect against confusing "who handles X?"
    # with "what is X's policy?"
    # -------------------------------------------------

    policy_terms = [
        "refund policy",
        "return policy",
        "cancellation policy",
        "privacy policy",
        "terms and conditions",
    ]

    handling_terms = [
        "who handles",
        "who can help",
        "who should I contact",
        "which team",
        "what team",
        "who deals with",
    ]

    if any(
        term in query
        for term in policy_terms
    ):

        if not any(
            term in query
            for term in handling_terms
        ):

            return False

    query_embedding = get_embedding(
        user_query
    )

    best_score = 0.0

    for supported_question in answerable_for:

        supported_embedding = get_embedding(
            supported_question
        )

        score = cosine_similarity(
            query_embedding,
            supported_embedding,
        )

        if score > best_score:
            best_score = score

    return best_score >= ANSWERABILITY_THRESHOLD


def filter_answerable_knowledge(
    user_query: str,
    knowledge: list[dict],
) -> list[dict]:
    """
    Keep only knowledge entries that are both
    semantically relevant and explicitly answerable.
    """

    answerable = []

    for item in knowledge:

        if retrieval_is_answerable(
            user_query,
            item,
        ):

            answerable.append(item)

    return answerable

def handle_tool_request(user_input: str) -> str | None:
    """
    Detect and execute a supported tool request.

    Returns:
        The tool result as a string, or None if no tool should handle
        the message.
    """
    selected_tool = select_tool(user_input)

    if not selected_tool:
        return None

    tool = get_tool(selected_tool)

    if tool is None:
        return None

    if selected_tool == "calculator":
        expression = extract_calculation(user_input)

        if expression is None:
            return "I couldn't find a valid calculation in that request."

        try:
            return tool["function"](expression)
        except (TypeError, ValueError):
            return "I couldn't calculate that safely."

    return None


def main():

    print(
        "Local AI Assistant with Smart Routing"
    )

    print("Type 'exit' to quit.")

    print("-" * 56)

    system_prompt = (
        "You are a helpful local assistant. "
        "Be concise, friendly, and practical. "
        "Do not claim to have internet access "
        "or cloud capabilities. "
        "Keep responses clear and useful."
    )

    print("Loading Laya classifier...")

    print(
        "Building semantic knowledge index..."
    )

    knowledge_index = build_index()

    print(
        f"Loaded {len(knowledge_index)} "
        f"knowledge chunks."
    )

    memory = MemoryManager(
        max_turns=10
    )

    while True:

        user_input = input(
            "\nYou: "
        ).strip()

        if user_input.lower() in {
            "exit",
            "quit",
            "bye",
        }:

            print("Goodbye!")

            break

        if not user_input:
            continue

        tool_reply = handle_tool_request(user_input)

        if tool_reply is not None:
            print(f"\nAssistant: {tool_reply}")
            continue

        memory.remember_user(
            user_input
        )

        try:

            classification = classify_message(
                user_input
            )

            classification = (
                apply_department_override(
                    user_input,
                    classification,
                )
            )


            decision = route_decision(
                classification
            )

            human_request_terms = (
                "need a human",
                "want a human",
                "speak to a human",
                "talk to a human",
                "human agent",
                "live agent",
                "support agent",
                "human representative",
            )

            if not any(
                term in user_input.lower()
                for term in human_request_terms
            ):
                decision = {
                    "action": "chat",
                    "reason": "No explicit human assistance requested.",
                }




            if (
                decision["action"]
                == "escalate_human"
            ):

                assistant_reply = (
                    "This request may need human assistance "
                    f"from the {classification['department']} team. "
                    "I cannot contact a human support agent "
                    "automatically. Please contact the appropriate "
                    "support team directly."
                )

                print(
                    f"\nAssistant: {assistant_reply}"
                )

                memory.remember_assistant(
                    assistant_reply
                )

                continue

            knowledge = search(
                user_input,
                knowledge_index,
                top_k=3,
                min_similarity=RAG_MIN_SIMILARITY,
            )

            answerable_knowledge = (
                filter_answerable_knowledge(
                    user_input,
                    knowledge,
                )
            )

            if answerable_knowledge:

                knowledge_context = (
                    "\n\n".join(
                        (
                            f"Knowledge:\n"
                            f"{item['content']}\n"
                            f"Semantic similarity: "
                            f"{item['score']:.3f}"
                        )
                        for item in answerable_knowledge
                    )
                )

            else:

                knowledge_context = ""

            conversation_context = format_memory(
                memory.get_context()
            )

            if knowledge_context:

                knowledge_section = (
                    "\n\n"
                    "RETRIEVED KNOWLEDGE:\n"
                    + knowledge_context
                )

            else:

                knowledge_section = (
                    "\n\n"
                    "RETRIEVED KNOWLEDGE:\n"
                    "No answerable knowledge was found."
                )

            reply = ask_model(
                user_input,
                system_prompt=(
                    system_prompt
                    + "\n\n"
                    + "Use the previous conversation "
                    + "to understand context and "
                    + "answer follow-up questions."
                    + "\n\n"
                    + "Use retrieved knowledge only "
                    + "when it explicitly answers "
                    + "the user's question."
                    + "\n\n"
                    + "Semantic similarity alone does "
                    + "not prove that knowledge contains "
                    + "the answer."
                    + "\n\n"
                    + "Do not invent factual information."
                    + "\n\n"
                    + "If the answer can be found "
                    + "in the previous conversation, "
                    + "you may use that information."
                    + "\n\n"
                    + "If neither the conversation "
                    + "memory nor the retrieved "
                    + "knowledge contains enough "
                    + "information to answer the "
                    + "question, say exactly: "
                    + "I don't have that information "
                    + "in my local knowledge base."
                    + "\n\n"
                    + "PREVIOUS CONVERSATION:\n"
                    + conversation_context
                    + knowledge_section
                ),
            )

            print(
                f"\nAssistant: {reply}"
            )

            memory.remember_assistant(
                reply
            )

        except Exception as exc:

            print(
                "\nError:",
                exc,
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()