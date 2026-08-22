"""
Knowledge-base document chunking.

This module converts loaded Markdown documents into smaller retrieval units.

The chunking strategy is heading-aware rather than blindly splitting text
into fixed-size blocks. Preserving Markdown headings is important because
the final customer-facing response must provide a source containing both
the original filename and the relevant heading.

Each generated chunk also retains the complete front-matter metadata from
its source document. This allows the retrieval layer to distinguish
authoritative active policies from superseded, draft, or internal content.
"""

import re
from dataclasses import dataclass
from typing import Any

from app.rag.document_loader import KnowledgeDocument


@dataclass
class DocumentChunk:
    """
    Represents a retrieval-ready section of a knowledge-base document.

    Attributes:
        chunk_id:
            Deterministic identifier generated from the source filename,
            heading, and chunk position.

        content:
            Text that will be embedded and stored in the vector index.

        filename:
            Original source Markdown filename.

        heading:
            Most specific Markdown heading associated with this chunk.

        metadata:
            Front-matter metadata copied from the original document.
    """

    chunk_id: str
    content: str
    filename: str
    heading: str
    metadata: dict[str, Any]


def _extract_sections(document: KnowledgeDocument) -> list[tuple[str, str]]:
    """
    Split a Markdown document into heading-aware sections.

    The function supports Markdown headings from H1 through H6.

    The H1 document title is treated as the initial context. When a more
    specific heading appears, subsequent content belongs to that heading.

    Returns:
        A list containing `(heading, content)` tuples.
    """

    lines = document.content.splitlines()

    sections: list[tuple[str, str]] = []

    current_heading = document.metadata.get(
        "title",
        document.filename,
    )

    current_lines: list[str] = []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    for line in lines:
        match = heading_pattern.match(line)

        if match:
            # Save the content accumulated under the previous heading before
            # starting a new section.
            if current_lines:
                section_content = "\n".join(current_lines).strip()

                if section_content:
                    sections.append(
                        (
                            current_heading,
                            section_content,
                        )
                    )

            current_heading = match.group(2).strip()
            current_lines = []

        else:
            current_lines.append(line)

    # Flush the final section after processing the complete document.
    if current_lines:
        section_content = "\n".join(current_lines).strip()

        if section_content:
            sections.append(
                (
                    current_heading,
                    section_content,
                )
            )

    return sections


def _split_large_section(
    content: str,
    max_characters: int,
    overlap_characters: int,
) -> list[str]:
    """
    Split an oversized section into overlapping text chunks.

    Heading-aware splitting is preferred, but some sections may still be too
    large for efficient embedding/retrieval. This fallback keeps chunks
    reasonably sized while retaining a small overlap between adjacent chunks.

    Character-based splitting is intentionally simple for this small corpus.
    We can replace it with token-based splitting later if evaluation shows
    that token boundaries materially improve retrieval quality.
    """

    if len(content) <= max_characters:
        return [content]

    chunks: list[str] = []

    start = 0
    content_length = len(content)

    while start < content_length:
        end = min(
            start + max_characters,
            content_length,
        )

        chunk = content[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop once the end of the section has been reached.
        if end >= content_length:
            break

        # Overlap allows information near a boundary to remain available
        # in the following chunk.
        start = max(
            end - overlap_characters,
            start + 1,
        )

    return chunks


def chunk_document(
    document: KnowledgeDocument,
    max_characters: int = 1500,
    overlap_characters: int = 200,
) -> list[DocumentChunk]:
    """
    Convert one knowledge document into retrieval-ready chunks.

    Args:
        document:
            Parsed Markdown document.

        max_characters:
            Maximum approximate character count for one chunk.

        overlap_characters:
            Number of characters shared between adjacent chunks when a
            section requires further splitting.

    Returns:
        A list of DocumentChunk objects.

    Raises:
        ValueError:
            If the chunk configuration is invalid.
    """

    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")

    if overlap_characters < 0:
        raise ValueError("overlap_characters cannot be negative")

    if overlap_characters >= max_characters:
        raise ValueError(
            "overlap_characters must be smaller than max_characters"
        )

    sections = _extract_sections(document)

    chunks: list[DocumentChunk] = []

    chunk_position = 0

    for heading, section_content in sections:
        section_chunks = _split_large_section(
            content=section_content,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )

        for section_chunk in section_chunks:
            chunk_position += 1

            # Include the heading in the embedded text. This gives the
            # embedding model additional context about what the passage is
            # describing and improves retrieval for heading-specific queries.
            searchable_content = (
                f"Heading: {heading}\n\n"
                f"{section_chunk}"
            )

            chunk_id = (
                f"{document.filename}:"
                f"{chunk_position:03d}"
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=searchable_content,
                    filename=document.filename,
                    heading=heading,
                    metadata=document.metadata.copy(),
                )
            )

    return chunks


def chunk_documents(
    documents: list[KnowledgeDocument],
    max_characters: int = 1500,
    overlap_characters: int = 200,
) -> list[DocumentChunk]:
    """
    Chunk an entire collection of knowledge-base documents.

    Documents are processed in their supplied order, while each document's
    internal chunk numbering starts independently.
    """

    all_chunks: list[DocumentChunk] = []

    for document in documents:
        all_chunks.extend(
            chunk_document(
                document=document,
                max_characters=max_characters,
                overlap_characters=overlap_characters,
            )
        )

    if not all_chunks:
        raise ValueError(
            "No chunks were generated from the knowledge base"
        )

    return all_chunks