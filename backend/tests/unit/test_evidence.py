"""
Tests for deterministic evidence selection.

These tests verify that:

    1. Weakly relevant authoritative passages are excluded.
    2. Relevant authoritative passages are retained.
    3. Ineligible sources are never selected.
    4. The number of evidence passages is limited.
    5. The default relevance threshold is enforced.
    6. Query-aware semantic similarity is used.
    7. Lexical/topic overlap is used as an additional relevance signal.
    8. An authoritative but semantically unrelated passage is rejected.
    9. An authoritative but topic-mismatched passage is rejected.

The embedding model is mocked in query-aware tests so these tests remain
fast, deterministic, and independent of the local Hugging Face model.
"""

from unittest.mock import patch

from app.models.retrieval import RetrievedChunk
from app.rag.evidence import select_evidence
from app.rag.reranker import RerankedResult


def make_result(
    filename: str,
    relevance: float,
    eligible: bool = True,
    content: str = "Example content",
    heading: str = "Example heading",
) -> RerankedResult:
    """
    Create a deterministic reranked result for testing.

    Args:
        filename:
            Source document filename.

        relevance:
            FAISS-derived relevance score.

        eligible:
            Whether the reranker considers the source safe for
            customer-facing use.

        content:
            Passage content used by query-aware lexical tests.

        heading:
            Passage heading used by query-aware semantic and lexical tests.

    Returns:
        A RerankedResult suitable for evidence-selection tests.
    """

    chunk = RetrievedChunk(
        chunk_id=f"{filename}:001",
        content=content,
        filename=filename,
        heading=heading,
        score=1.0,
        metadata={},
    )

    return RerankedResult(
        chunk=chunk,
        authority_score=90,
        relevance_score=relevance,
        combined_score=relevance,
        eligible=eligible,
        reason="active, customer-facing, official",
    )


def test_low_relevance_authoritative_result_is_excluded() -> None:
    """
    Authority alone should not make an unrelated document evidence.
    """

    result = make_result(
        filename="07-warranty.md",
        relevance=0.20,
    )

    selected = select_evidence(
        [result],
        minimum_relevance=0.40,
    )

    assert selected == []


def test_relevant_authoritative_result_is_selected() -> None:
    """
    Relevant authoritative evidence should be retained.
    """

    result = make_result(
        filename="01-returns-policy-current.md",
        relevance=0.49,
    )

    selected = select_evidence(
        [result],
        minimum_relevance=0.40,
    )

    assert len(selected) == 1

    assert (
        selected[0].chunk.filename
        == "01-returns-policy-current.md"
    )


def test_ineligible_result_is_never_selected() -> None:
    """
    Superseded or unsafe sources must never enter final evidence.
    """

    result = make_result(
        filename="02-returns-policy-legacy.md",
        relevance=0.90,
        eligible=False,
    )

    selected = select_evidence(
        [result]
    )

    assert selected == []


def test_evidence_count_is_limited() -> None:
    """
    The selector should never return more than max_results.
    """

    results = [
        make_result(
            filename=f"policy-{index}.md",
            relevance=0.80,
        )
        for index in range(5)
    ]

    selected = select_evidence(
        results,
        max_results=3,
    )

    assert len(selected) == 3


def test_default_threshold_rejects_weak_authoritative_evidence() -> None:
    """
    The default evidence policy should reject passages whose semantic
    relevance is below the configured evidence threshold.
    """

    result = make_result(
        filename="07-warranty.md",
        relevance=0.46,
    )

    selected = select_evidence(
        [result]
    )

    assert selected == []


def test_default_threshold_accepts_strong_authoritative_evidence() -> None:
    """
    Strongly relevant authoritative evidence should pass the default
    evidence threshold.
    """

    result = make_result(
        filename="01-returns-policy-current.md",
        relevance=0.49,
    )

    selected = select_evidence(
        [result]
    )

    assert len(selected) == 1

    assert (
        selected[0].chunk.filename
        == "01-returns-policy-current.md"
    )


def test_query_aware_selection_uses_query_similarity() -> None:
    """
    Query-aware evidence selection should use semantic similarity between
    the customer question and the actual passage.

    The embedding model and similarity calculation are mocked so this test
    remains deterministic and does not load the Sentence Transformer model.
    """

    relevant = make_result(
        filename="01-returns-policy-current.md",
        relevance=0.49,
        heading="Standard return window",
        content=(
            "Customers may return unused items within "
            "the standard return window."
        ),
    )

    unrelated = make_result(
        filename="07-warranty.md",
        relevance=0.49,
        heading="Warranty periods",
        content=(
            "Warranty coverage applies to manufacturing defects."
        ),
    )

    class FakeEmbeddingModel:
        """
        Minimal fake embedding model used by the unit test.
        """

        def embed_query(
            self,
            text: str,
        ) -> list[float]:
            return [1.0, 0.0]

    def fake_query_similarity(
        query_embedding: list[float],
        result: RerankedResult,
    ) -> float:
        """
        Return deterministic semantic similarity values.

        The returns policy is intentionally highly similar to the query,
        while the warranty document is intentionally unrelated.
        """

        if (
            result.chunk.filename
            == "01-returns-policy-current.md"
        ):
            return 0.90

        return 0.10

    with patch(
        "app.rag.evidence.get_embedding_model",
        return_value=FakeEmbeddingModel(),
    ), patch(
        "app.rag.evidence._calculate_query_similarity",
        side_effect=fake_query_similarity,
    ), patch(
        "app.rag.evidence._calculate_heading_similarity",
        return_value=0.50,
    ):
        selected = select_evidence(
            [unrelated, relevant],
            query=(
                "How long can I return "
                "an unused backpack?"
            ),
            minimum_query_similarity=0.35,
            minimum_lexical_similarity=0.10,
        )

    assert len(selected) == 1

    assert (
        selected[0].chunk.filename
        == "01-returns-policy-current.md"
    )


def test_query_aware_selection_rejects_semantically_unrelated_source() -> None:
    """
    An authoritative passage must still be rejected when it is not
    sufficiently similar to the actual customer query.

    This protects the LLM from receiving authoritative but unrelated
    evidence.
    """

    warranty = make_result(
        filename="07-warranty.md",
        relevance=0.50,
        heading="Warranty periods",
        content=(
            "Warranty coverage applies to manufacturing defects."
        ),
    )

    class FakeEmbeddingModel:
        """
        Minimal fake embedding model used by the unit test.
        """

        def embed_query(
            self,
            text: str,
        ) -> list[float]:
            return [1.0, 0.0]

    with patch(
        "app.rag.evidence.get_embedding_model",
        return_value=FakeEmbeddingModel(),
    ), patch(
        "app.rag.evidence._calculate_query_similarity",
        return_value=0.10,
    ):
        selected = select_evidence(
            [warranty],
            query=(
                "How long can I return "
                "an unused backpack?"
            ),
            minimum_query_similarity=0.35,
        )

    assert selected == []


def test_topic_mismatched_authoritative_source_is_rejected() -> None:
    """
    An authoritative document should not become evidence merely because
    its embedding is semantically related to the customer question.

    This regression test protects against the observed failure where the
    warranty policy outranked the returns policy for a return-window query.
    """

    warranty = make_result(
        filename="07-warranty.md",
        relevance=0.515,
        heading="Warranty periods",
        content=(
            "Coverage applies to manufacturing defects "
            "and warranty claims."
        ),
    )

    returns = make_result(
        filename="01-returns-policy-current.md",
        relevance=0.491,
        heading="Standard return window",
        content=(
            "Customers may return unused items within "
            "the standard return window."
        ),
    )

    class FakeEmbeddingModel:
        """
        Minimal fake embedding model used by the regression test.
        """

        def embed_query(
            self,
            text: str,
        ) -> list[float]:
            return [1.0, 0.0]

    def fake_query_similarity(
        query_embedding: list[float],
        result: RerankedResult,
    ) -> float:
        """
        Simulate the semantic similarity values observed during debugging.

        Both documents are semantically related enough to pass the semantic
        threshold. Lexical/topic overlap must therefore distinguish them.
        """

        return {
            "07-warranty.md": 0.486,
            "01-returns-policy-current.md": 0.474,
        }[result.chunk.filename]

    with patch(
        "app.rag.evidence.get_embedding_model",
        return_value=FakeEmbeddingModel(),
    ), patch(
        "app.rag.evidence._calculate_query_similarity",
        side_effect=fake_query_similarity,
    ), patch(
        "app.rag.evidence._calculate_heading_similarity",
        return_value=0.20,
    ):
        selected = select_evidence(
            [warranty, returns],
            query=(
                "How long does a regular customer "
                "have to return an unused backpack?"
            ),
            minimum_query_similarity=0.30,
            minimum_lexical_similarity=0.10,
        )

    selected_filenames = [
        result.chunk.filename
        for result in selected
    ]

    assert (
        "01-returns-policy-current.md"
        in selected_filenames
    )

    assert (
        "07-warranty.md"
        not in selected_filenames
    )


def test_lexical_similarity_can_match_plural_and_singular_forms() -> None:
    """
    Lexical normalization should recognize common morphological variants.

    For example:

        return
        returns

    should contribute to the same topic overlap.
    """

    result = make_result(
        filename="01-returns-policy-current.md",
        relevance=0.50,
        heading="Standard returns",
        content=(
            "Customers can return unused products."
        ),
    )

    class FakeEmbeddingModel:
        """
        Minimal fake embedding model.
        """

        def embed_query(
            self,
            text: str,
        ) -> list[float]:
            return [1.0, 0.0]

    with patch(
        "app.rag.evidence.get_embedding_model",
        return_value=FakeEmbeddingModel(),
    ), patch(
        "app.rag.evidence._calculate_query_similarity",
        return_value=0.80,
    ), patch(
        "app.rag.evidence._calculate_heading_similarity",
        return_value=0.50,
    ):
        selected = select_evidence(
            [result],
            query="What is the return window?",
            minimum_query_similarity=0.30,
            minimum_lexical_similarity=0.10,
        )

    assert len(selected) == 1

    assert (
        selected[0].chunk.filename
        == "01-returns-policy-current.md"
    )