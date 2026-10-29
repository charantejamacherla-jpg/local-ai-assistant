from vector_search import build_index, search


TEST_QUERIES = [
    "When are you open?",
    "Are you closed on public holidays?",
    "How much do your plans cost?",
    "Who handles technical problems?",
    "Who can help me with my account?",
    "What is your refund policy?",
    "Do you offer student discounts?",
]


def main():
    print("Building semantic knowledge index...")
    index = build_index()

    print(f"Loaded {len(index)} knowledge chunks.")
    print("-" * 70)

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")

        results = search(
            query,
            index,
            top_k=3,
        )

        for result in results:
            print(
                f"  Score: {result['score']:.3f} | "
                f"Chunk: {result['chunk_id']} | "
                f"{result['content'].splitlines()[0]}"
            )


if __name__ == "__main__":
    main()