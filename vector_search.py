from document_loader import load_documents
from chunker import chunk_document
from embeddings import get_embedding, cosine_similarity
from knowledge_base import KNOWLEDGE_BASE


def get_answerable_for(
    chunk_content: str,
) -> list[str]:
    """
    Find the knowledge-base metadata that belongs
    to this document chunk.
    """

    for entry in KNOWLEDGE_BASE:

        topic = entry["topic"]

        if topic.lower() in chunk_content.lower():

            return entry.get(
                "answerable_for",
                [],
            )

    return []


def build_index() -> list[dict]:
    """
    Load documents, split them into chunks,
    and create an embedding for each chunk.
    """

    documents = load_documents()

    index = []

    for document in documents:

        chunks = chunk_document(
            document
        )

        for chunk in chunks:

            embedding = get_embedding(
                chunk["content"]
            )

            answerable_for = (
                get_answerable_for(
                    chunk["content"]
                )
            )

            index.append(
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "embedding": embedding,
                    "answerable_for": answerable_for,
                }
            )

    return index


def search(
    query: str,
    index: list[dict],
    top_k: int = 3,
    min_similarity: float = 0.55,
) -> list[dict]:
    """
    Find the most semantically similar chunks.

    Results below min_similarity are rejected.
    """

    query_embedding = get_embedding(
        query
    )

    results = []

    for item in index:

        score = cosine_similarity(
            query_embedding,
            item["embedding"],
        )

        if score >= min_similarity:

            results.append(
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "content": item["content"],
                    "score": score,
                    "answerable_for": item.get(
                        "answerable_for",
                        [],
                    ),
                }
            )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]