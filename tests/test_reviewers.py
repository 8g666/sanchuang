from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.reviewers.semantic_reviewer import SemanticReviewer
from docguard_rag.server.reviewers.typo_reviewer import TypoReviewer


def test_typo_reviewer_returns_absolute_location():
    block = DocumentBlock(
        block_id="block_1",
        document_id="doc_1",
        block_type="sentence",
        text="系统能够被直接掉用。",
        start_offset=20,
        paragraph_index=1,
        sentence_index=0,
    )

    issue = TypoReviewer().review(block, "review_1", "TYPO")[0]

    assert issue.problem_text == "掉用"
    assert issue.suggestion == "调用"
    assert issue.location.start_offset == 27
    assert issue.location.end_offset == 29


def test_semantic_reviewer_detects_redundant_connective():
    block = DocumentBlock(
        block_id="block_2",
        document_id="doc_1",
        block_type="sentence",
        text="该方法在提高性能后，因此能够降低检索时间。",
    )

    issues = SemanticReviewer().review(block, "review_1", "SEMANTIC_UNCLEAR")

    assert len(issues) == 1
    assert issues[0].issue_type == "SEMANTIC_UNCLEAR"
    assert "后，" in issues[0].suggestion
