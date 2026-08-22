"""
Knowledge-base document loader.

This module loads Markdown files from the supplied knowledge-base directory
and separates document metadata from the main Markdown content.

The original source files are never modified. The loader creates an internal
representation that can later be chunked, embedded, and indexed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class KnowledgeDocument:
    """
    Internal representation of a knowledge-base Markdown document.

    Attributes:
        filename: Original source filename.
        content: Markdown content excluding front matter.
        metadata: Front-matter fields preserved from the source document.
    """

    filename: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_markdown_file(file_path: Path) -> KnowledgeDocument:
    """
    Parse a single Markdown file.

    The repository uses Markdown documents that may contain YAML front matter.
    When front matter is present, it is parsed into a metadata dictionary.
    """

    raw_content = file_path.read_text(encoding="utf-8")

    metadata: dict[str, Any] = {}
    content = raw_content

    # YAML front matter is conventionally delimited by --- at the beginning
    # and end of the metadata section.
    if raw_content.startswith("---"):
        parts = raw_content.split("---", 2)

        if len(parts) == 3:
            _, raw_metadata, content = parts

            parsed_metadata = yaml.safe_load(raw_metadata)

            if isinstance(parsed_metadata, dict):
                metadata = parsed_metadata

    return KnowledgeDocument(
        filename=file_path.name,
        content=content.strip(),
        metadata=metadata,
    )


def load_knowledge_base(knowledge_base_path: str | Path) -> list[KnowledgeDocument]:
    """
    Load all Markdown documents from the knowledge-base directory.

    Files are processed in deterministic filename order so that indexing
    behavior is reproducible across runs.
    """

    directory = Path(knowledge_base_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory does not exist: {directory}"
        )

    documents: list[KnowledgeDocument] = []

    for file_path in sorted(directory.glob("*.md")):
        documents.append(parse_markdown_file(file_path))

    if not documents:
        raise ValueError(
            f"No Markdown documents found in knowledge base: {directory}"
        )

    return documents