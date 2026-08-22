"""
Schemas used by the knowledge-base retrieval pipeline.

The retrieval layer preserves both the document content and its source
metadata. Metadata is important for this project because semantic similarity
alone is not sufficient to determine which document should be trusted.

For example, the repository contains:
    - active policies
    - superseded policies
    - internal documents
    - draft documents
    - product information

The metadata allows the application to apply document-precedence and
customer-visibility rules after vector retrieval.
"""

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    Represents one passage retrieved from the knowledge base.

    Attributes:
        chunk_id:
            Stable identifier for the indexed chunk.

        content:
            Text that was retrieved from the source document.

        filename:
            Original Markdown filename. This is required for source citations.

        heading:
            Markdown heading associated with the passage. This is required
            for precise source references in customer-facing answers.

        score:
            Similarity score returned by the vector search.

        metadata:
            Original front-matter metadata from the source document.
            Metadata is preserved rather than flattened so that future
            document fields can be introduced without changing this schema.
    """

    chunk_id: str

    content: str

    filename: str

    heading: str

    score: float

    metadata: dict[str, Any] = Field(default_factory=dict)