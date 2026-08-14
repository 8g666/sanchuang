import json
from uuid import uuid4

from sqlalchemy.orm import Session

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.core.provider_factory import create_llm_provider
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.issue import DocumentIssue
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.repositories.review_repository import ReviewRepository
from docguard_rag.server.reviewers.base import DocumentReviewer
from docguard_rag.server.reviewers.llm_reviewer import LLMReviewer
from docguard_rag.server.reviewers.semantic_reviewer import SemanticReviewer
from docguard_rag.server.reviewers.typo_reviewer import TypoReviewer


class ReviewService:
    supported_types = {
        "TYPO",
        "FORMAT_ERROR",
        "GRAMMAR",
        "SEMANTIC_UNCLEAR",
        "CONTEXT_BREAK",
        "TERM_INCONSISTENT",
        "FACT_CONFLICT",
    }

    def __init__(self, reviewers: list[DocumentReviewer] | None = None):
        if reviewers is None:
            reviewers = [TypoReviewer(), SemanticReviewer()]
            if get_settings().llm_review_enabled:
                provider = create_llm_provider()
                if provider is not None:
                    reviewers.append(LLMReviewer(provider))
        self.reviewers = reviewers
        self.document_repository = DocumentRepository()
        self.review_repository = ReviewRepository()

    def create_review(
        self,
        db: Session,
        document_ids: list[str],
        check_types: list[str],
        mode: str,
        tenant_id: str = "default",
    ) -> tuple[str, list[DocumentIssue]]:
        review_id, normalized_types = self.prepare_review(
            db,
            document_ids,
            check_types,
            mode,
            tenant_id=tenant_id,
        )
        issues = self.run_review(
            db,
            review_id,
            document_ids,
            normalized_types,
            mode,
        )
        return review_id, issues

    def prepare_review(
        self,
        db: Session,
        document_ids: list[str],
        check_types: list[str],
        mode: str,
        tenant_id: str = "default",
        review_id: str | None = None,
    ) -> tuple[str, list[str]]:
        review_id = review_id or f"review_{uuid4().hex[:12]}"
        normalized_types = list(dict.fromkeys(value.upper() for value in check_types))
        self.review_repository.create_review(
            db, review_id, document_ids, mode, normalized_types, tenant_id
        )
        return review_id, normalized_types

    def run_review(
        self,
        db: Session,
        review_id: str,
        document_ids: list[str],
        normalized_types: list[str],
        mode: str,
    ) -> list[DocumentIssue]:
        threshold_by_mode = {"FAST": 0.9, "BALANCED": 0.75, "STRICT": 0.6}
        threshold = threshold_by_mode[mode]
        issues: list[DocumentIssue] = []
        for document_id in document_ids:
            records = self.document_repository.get_blocks(db, document_id)
            sentence_records = [record for record in records if record.block_type == "sentence"]
            paragraph_records = [record for record in records if record.block_type == "paragraph"]
            review_records = paragraph_records + sentence_records
            for record in review_records:
                block = DocumentBlock(
                    block_id=record.block_id,
                    document_id=record.document_id,
                    block_type=record.block_type,
                    text=record.text,
                    page_number=record.page_number,
                    section_path=json.loads(record.section_path or "[]"),
                    paragraph_index=record.paragraph_index,
                    sentence_index=record.sentence_index,
                    start_offset=record.start_offset,
                    end_offset=record.end_offset,
                )
                for check_type in normalized_types:
                    for reviewer in self.reviewers:
                        if check_type not in reviewer.issue_types:
                            continue
                        if isinstance(reviewer, LLMReviewer) and (
                            mode == "FAST" or record.block_type != "sentence"
                        ):
                            continue
                        issues.extend(
                            issue
                            for issue in reviewer.review(block, review_id, check_type)
                            if issue.confidence >= threshold
                        )

        # A TYPO review also checks repeated punctuation, but de-duplicate exact spans.
        unique = {
            (
                issue.document_id,
                issue.issue_type,
                issue.location.start_offset,
                issue.location.end_offset,
                issue.problem_text,
            ): issue
            for issue in issues
        }
        persisted = list(unique.values())
        self.review_repository.save_issues(db, persisted)
        self.review_repository.complete_review(db, review_id)
        return persisted
