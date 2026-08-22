"""
Knowledge-base retrieval service.

This module performs semantic candidate retrieval using the FAISS index.

Important architectural decision:
FAISS similarity is treated as candidate retrieval, not as the final
authority decision.

The repository intentionally contains superseded, internal, draft, and
conflicting documents. Therefore, a later metadata-aware reranking layer
will determine which retrieved candidates are appropriate for customer
answers.

This separation prevents vector similarity from silently becoming a
business-policy decision.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.models.retrieval import RetrievedChunk
from app.rag.index import load_faiss_index


class KnowledgeRetriever:
    """
    Semantic retrieval interface for the Aster & Row knowledge base.
    """

    def __init__(
        self,
        vectorstore: FAISS,
    ) -> None:
        """
        Initialize the retriever with an existing FAISS vector store.
        """

        self.vectorstore = vectorstore

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """
        Retrieve semantically similar knowledge-base passages.

        Args:
            query:
                User question or retrieval query.

            top_k:
                Maximum number of candidate passages to retrieve.

        Returns:
            Retrieved chunks ordered by vector similarity.

        Note:
            These results are candidates only. They must pass through the
            metadata-aware precedence layer before being used as authoritative
            customer-facing evidence.
        """

        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
        )

        retrieved: list[RetrievedChunk] = []

        for document, score in results:
            metadata = document.metadata

            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(
                        metadata.get(
                            "chunk_id",
                            "",
                        )
                    ),
                    content=document.page_content,
                    filename=str(
                        metadata.get(
                            "filename",
                            "unknown",
                        )
                    ),
                    heading=str(
                        metadata.get(
                            "heading",
                            "Unknown section",
                        )
                    ),
                    score=float(score),
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if key not in {
                            "chunk_id",
                            "filename",
                            "heading",
                        }
                    },
                )
            )

        return retrieved


def create_retriever() -> KnowledgeRetriever:
    """
    Load the configured FAISS index and create a retriever.

    Keeping index loading separate from the class allows the retriever to be
    dependency-injected during unit tests.
    """

    index_path = Path(
        settings.faiss_index_path
    )

    if not index_path.is_absolute():
        index_path = (
            Path(__file__).resolve().parents[2]
            / index_path
        )

    vectorstore = load_faiss_index(index_path)

    return KnowledgeRetriever(vectorstore)