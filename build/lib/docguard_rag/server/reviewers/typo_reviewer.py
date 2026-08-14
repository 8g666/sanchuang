import json
import re
from uuid import uuid4

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.issue import DocumentIssue, IssueLocation
from docguard_rag.server.reviewers.base import DocumentReviewer


class TypoReviewer(DocumentReviewer):
    """High-precision typo recall using a configurable confusion dictionary."""

    issue_types = {"TYPO", "FORMAT_ERROR"}

    def __init__(self, confusion_map: dict[str, str] | None = None):
        defaults = {
            "掉用": "调用",
            "登路": "登录",
            "布署": "部署",
            "按装": "安装",
            "帐号": "账号",
            "备分": "备份",
            "必需品": "必需品",  # Explicitly retained to avoid substring over-correction.
        }
        if confusion_map is None:
            try:
                configured = json.loads(get_settings().typo_confusion_json or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("DOCGUARD_TYPO_CONFUSION_JSON must be valid JSON") from exc
            if not isinstance(configured, dict):
                raise ValueError("DOCGUARD_TYPO_CONFUSION_JSON must be a JSON object")
            defaults.update(
                {str(original): str(replacement) for original, replacement in configured.items()}
            )
        self.confusion_map = confusion_map or defaults

    def review(
        self,
        block: DocumentBlock,
        review_id: str,
        requested_type: str = "TYPO",
    ) -> list[DocumentIssue]:
        issues: list[DocumentIssue] = []
        for original, replacement in self.confusion_map.items():
            if original == replacement:
                continue
            for match in re.finditer(re.escape(original), block.text):
                issues.append(
                    self._issue(
                        block,
                        review_id,
                        "TYPO",
                        original,
                        replacement,
                        "命中预置的高置信度常见错别字词表。",
                        match.start(),
                        match.end(),
                        0.97,
                    )
                )

        for match in re.finditer(r"([，。！？；：、])\1+", block.text):
            mark = match.group(1)
            issues.append(
                self._issue(
                    block,
                    review_id,
                    "FORMAT_ERROR",
                    match.group(0),
                    mark,
                    "相同的中文标点连续重复。",
                    match.start(),
                    match.end(),
                    0.99,
                )
            )
        return [issue for issue in issues if requested_type in {issue.issue_type, "TYPO"}]

    @staticmethod
    def _issue(
        block: DocumentBlock,
        review_id: str,
        issue_type: str,
        problem_text: str,
        suggestion: str,
        reason: str,
        relative_start: int,
        relative_end: int,
        confidence: float,
    ) -> DocumentIssue:
        start = (
            block.start_offset + relative_start
            if block.start_offset is not None
            else relative_start
        )
        return DocumentIssue(
            issue_id=f"issue_{uuid4().hex[:12]}",
            document_id=block.document_id,
            review_id=review_id,
            issue_type=issue_type,
            severity="MEDIUM",
            original_text=block.text,
            problem_text=problem_text,
            suggestion=suggestion,
            reason=reason,
            confidence=confidence,
            location=IssueLocation(
                page=block.page_number,
                section_path=list(block.section_path),
                paragraph_index=block.paragraph_index,
                sentence_index=block.sentence_index,
                start_offset=start,
                end_offset=start + (relative_end - relative_start),
            ),
        )
