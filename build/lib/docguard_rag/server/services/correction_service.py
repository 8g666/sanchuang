import re

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.core.provider_factory import create_llm_provider
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.utils import new_block_id, sentence_blocks
from docguard_rag.server.reviewers.llm_reviewer import LLMReviewer
from docguard_rag.server.reviewers.semantic_reviewer import SemanticReviewer
from docguard_rag.server.reviewers.typo_reviewer import TypoReviewer
from docguard_rag.server.schemas.correction import (
    CorrectionIssue,
    CorrectionLocation,
    CorrectionRequest,
    CorrectionResult,
)


class CorrectionService:
    """Stateless correction pipeline for text supplied directly by callers."""

    def __init__(self, reviewers=None):
        if reviewers is None:
            reviewers = [TypoReviewer(), SemanticReviewer()]
            if get_settings().llm_review_enabled:
                provider = create_llm_provider()
                if provider is not None:
                    reviewers.append(LLMReviewer(provider))
        self.reviewers = reviewers

    def correct(self, payload: CorrectionRequest) -> CorrectionResult:
        threshold = {"FAST": 0.9, "BALANCED": 0.75, "STRICT": 0.6}[payload.mode]
        issues = []
        for block in self._blocks(payload.text):
            for check_type in dict.fromkeys(value.upper() for value in payload.check_types):
                for reviewer in self.reviewers:
                    if check_type not in reviewer.issue_types:
                        continue
                    if isinstance(reviewer, LLMReviewer) and (
                        payload.mode == "FAST" or block.block_type != "sentence"
                    ):
                        continue
                    issues.extend(
                        issue
                        for issue in reviewer.review(block, "inline", check_type)
                        if issue.confidence >= threshold
                    )

        unique = {
            (
                issue.issue_type,
                issue.location.start_offset,
                issue.location.end_offset,
                issue.problem_text,
            ): issue
            for issue in issues
        }
        ordered = sorted(
            unique.values(),
            key=lambda issue: (
                issue.location.start_offset if issue.location.start_offset is not None else -1,
                -issue.confidence,
            ),
        )
        corrected = self._apply(payload.text, ordered) if payload.apply_changes else payload.text
        return CorrectionResult(
            original_text=payload.text,
            corrected_text=corrected,
            changed=corrected != payload.text,
            issues=[
                CorrectionIssue(
                    issue_id=issue.issue_id,
                    issue_type=issue.issue_type,
                    severity=issue.severity,
                    original_text=issue.original_text,
                    problem_text=issue.problem_text,
                    suggestion=issue.suggestion,
                    reason=issue.reason,
                    confidence=issue.confidence,
                    location=CorrectionLocation(
                        paragraph_index=issue.location.paragraph_index,
                        sentence_index=issue.location.sentence_index,
                        start_offset=issue.location.start_offset,
                        end_offset=issue.location.end_offset,
                    ),
                )
                for issue in ordered
            ],
        )

    @staticmethod
    def _blocks(text: str) -> list[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        paragraph_index = 0
        for match in re.finditer(r"[^\r\n]+", text):
            raw = match.group(0)
            value = raw.strip()
            if not value:
                continue
            leading = len(raw) - len(raw.lstrip())
            start = match.start() + leading
            paragraph = DocumentBlock(
                block_id=new_block_id(),
                document_id="inline",
                block_type="paragraph",
                text=value,
                paragraph_index=paragraph_index,
                start_offset=start,
                end_offset=start + len(value),
            )
            blocks.append(paragraph)
            blocks.extend(sentence_blocks(paragraph))
            paragraph_index += 1
        return blocks

    @staticmethod
    def _apply(text: str, issues) -> str:
        selected = []
        occupied: list[tuple[int, int]] = []
        for issue in sorted(issues, key=lambda value: value.confidence, reverse=True):
            start = issue.location.start_offset
            end = issue.location.end_offset
            if start is None or end is None or start < 0 or end > len(text) or start >= end:
                continue
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            if text[start:end] != issue.problem_text:
                continue
            occupied.append((start, end))
            selected.append((start, end, issue.suggestion))

        corrected = text
        for start, end, suggestion in sorted(selected, reverse=True):
            corrected = corrected[:start] + suggestion + corrected[end:]
        return corrected
