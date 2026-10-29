def chunk_document(
    document: dict,
) -> list[dict]:
    """
    Split a document into paragraph-based chunks.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in document["content"].split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    for index, paragraph in enumerate(paragraphs):
        chunks.append(
            {
                "source": document["source"],
                "chunk_id": index,
                "content": paragraph,
            }
        )

    return chunks