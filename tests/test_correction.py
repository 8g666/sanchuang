import json

from docguard_rag import DocGuard
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.reviewers.llm_reviewer import LLMReviewer


class StructuredReviewProvider:
    def generate(self, messages, response_schema=None):
        return {
            "content": json.dumps(
                {
                    "issues": [
                        {
                            "problem_text": "进行优化",
                            "suggestion": "优化",
                            "reason": "表达冗余",
                            "confidence": 0.92,
                            "severity": "LOW",
                            "start_offset": 2,
                            "end_offset": 6,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        }


def test_direct_text_correction_api(tmp_path):
    with DocGuard(
        database_url=f"sqlite:///{tmp_path / 'correction.db'}",
        upload_dir=tmp_path / "uploads",
    ) as guard:
        result = guard.correct("系统能够被直接掉用。。")

    assert result.changed is True
    assert result.corrected_text == "系统能够被直接调用。"
    assert {issue.issue_type for issue in result.issues} >= {"TYPO", "FORMAT_ERROR"}


def test_llm_reviewer_validates_and_locates_suggestion():
    reviewer = LLMReviewer(StructuredReviewProvider())
    block = DocumentBlock(
        block_id="block",
        document_id="document",
        block_type="sentence",
        text="需要进行优化。",
        start_offset=10,
    )

    issues = reviewer.review(block, "review", "SEMANTIC_UNCLEAR")

    assert len(issues) == 1
    assert issues[0].suggestion == "优化"
    assert issues[0].location.start_offset == 12
