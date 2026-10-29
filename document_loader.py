from pathlib import Path


DOCUMENTS_DIR = Path("documents")


def load_documents() -> list[dict]:
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents