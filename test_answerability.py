
from app import (
    filter_answerable_knowledge,
)
from vector_search import build_index, search

index = build_index()

questions = [
    "Who handles technical problems?",
    "What is your refund policy?",
    "Do you offer student discounts?",
    "When are you open?",
]

for question in questions:
    print("\nQUESTION:", question)

    retrieved = search(
        question,
        index,
        top_k=3,
        min_similarity=0.55,
    )

    answerable = filter_answerable_knowledge(
        question,
        retrieved,
    )

    print("Retrieved:", len(retrieved))
    print("Answerable:", len(answerable))

    for item in answerable:
        print(
            "Accepted:",
            item.get("content", "")[:120],
        )
