"""
Build the Aster & Row FAISS knowledge-base index.

This script is intentionally separate from application startup. The source
documents should be parsed and indexed explicitly so that application startup
does not unexpectedly rebuild the vector database.

Usage:
    python scripts/build_index.py
"""

from pathlib import Path

from app.core.config import settings
from app.rag.chunker import chunk_documents
from app.rag.document_loader import load_knowledge_base
from app.rag.index import build_faiss_index, save_faiss_index


# Resolve paths relative to the repository root rather than the current
# terminal directory. This makes the script more predictable when executed
# from different working directories.
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge-base"

FAISS_PATH = BACKEND_DIR / settings.faiss_index_path


def main() -> None:
    """
    Load, chunk, embed, and persist the complete knowledge base.
    """

    print("Loading knowledge-base documents...")

    documents = load_knowledge_base(
        KNOWLEDGE_BASE_PATH
    )

    print(f"Loaded {len(documents)} documents.")

    print("Creating document chunks...")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Building FAISS index...")

    vectorstore = build_faiss_index(chunks)

    print("Saving FAISS index...")

    save_faiss_index(
        vectorstore,
        FAISS_PATH,
    )

    print(f"FAISS index saved to: {FAISS_PATH}")


if __name__ == "__main__":
    main()