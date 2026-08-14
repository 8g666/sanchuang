import re
from uuid import uuid4

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.issue import DocumentIssue, IssueLocation
from docguard_rag.server.reviewers.base import DocumentReviewer


class SemanticReviewer(DocumentReviewer):
    """Conservative rules for conspicuous grammar and semantic-connective defects."""

    issue_types = {"GRAMMAR", "SEMANTIC_UNCLEAR", "CONTEXT_BREAK"}
    patterns = (
        (
            re.compile(r"在(.{1,40}?)后，因此"),
            "SEMANTIC_UNCLEAR",
            lambda value: value.replace("后，因此", "后，"),
            "“在……后”已经表达承接关系，再使用“因此”会造成连接关系重复。",
            0.91,
        ),
        (
            re.compile(r"因为(.{1,60}?)所以因此"),
            "GRAMMAR",
            lambda value: value.replace("所以因此", "因此"),
            "“所以”和“因此”语义重复。",
            0.96,
        ),
        (
            re.compile(r"(?:进行进行|可以可以|以及以及|的的|了了)"),
            "GRAMMAR",
            lambda value: value[: len(value) // 2],
            "相邻词语发生明显重复。",
            0.96,
        ),
    )

    def review(
        self,
        block: DocumentBlock,
        review_id: str,
        requested_type: str = "SEMANTIC_UNCLEAR",
    ) -> list[DocumentIssue]:
        issues: list[DocumentIssue] = []
        for pattern, issue_type, replacement, reason, confidence in self.patterns:
            for match in pattern.finditer(block.text):
                if requested_type not in {issue_type, "SEMANTIC_UNCLEAR"}:
                    continue
                problem = match.group(0)
                issues.append(
                    self._issue(
                        block,
                        review_id,
                        issue_type,
                        problem,
                        replacement(problem),
                        reason,
                        match.start(),
                        match.end(),
                        confidence,
                    )
                )

        if requested_type in {"SEMANTIC_UNCLEAR", "GRAMMAR"} and len(block.text) > 180:
            issues.append(
                self._issue(
                    block,
                    review_id,
                    "SEMANTIC_UNCLEAR",
                    block.text,
                    "建议拆分长句，并明确各分句的主语与逻辑关系。",
                    "句子过长，可能增加指代或逻辑关系不清的风险，需人工确认。",
                    0,
                    len(block.text),
                    0.68,
                    severity="LOW",
                )
            )
        return issues

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
        severity: str = "MEDIUM",
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
            severity=severity,
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
