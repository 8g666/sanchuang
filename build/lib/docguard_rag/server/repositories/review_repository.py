import json
from datetime import datetime

from sqlalchemy.orm import Session

from docguard_rag.server.database.models import DocumentIssueTable, ReviewJobTable
from docguard_rag.server.models.issue import DocumentIssue


class ReviewRepository:
    def create_review(
        self,
        db: Session,
        review_id: str,
        document_ids: list[str],
        mode: str,
        check_types: list[str],
        tenant_id: str = "default",
    ) -> ReviewJobTable:
        record = ReviewJobTable(
            review_id=review_id,
            tenant_id=tenant_id,
            document_id=json.dumps(document_ids, ensure_ascii=False),
            status="PROCESSING",
            mode=mode,
            check_types=json.dumps(check_types),
            progress=0,
        )
        db.add(record)
        db.commit()
        return record

    def complete_review(self, db: Session, review_id: str) -> ReviewJobTable | None:
        record = self.get_review(db, review_id)
        if record:
            record.status = "COMPLETED"
            record.progress = 100
            record.completed_at = datetime.now()
            db.commit()
        return record

    def get_review(
        self,
        db: Session,
        review_id: str,
        tenant_id: str | None = None,
    ) -> ReviewJobTable | None:
        query = db.query(ReviewJobTable).filter(ReviewJobTable.review_id == review_id)
        if tenant_id is not None:
            query = query.filter(ReviewJobTable.tenant_id == tenant_id)
        return query.one_or_none()

    def fail_review(self, db: Session, review_id: str, error: str) -> ReviewJobTable | None:
        record = self.get_review(db, review_id)
        if record:
            record.status = "FAILED"
            record.error_message = error
            record.completed_at = datetime.now()
            db.commit()
        return record

    def get_issue(self, db: Session, issue_id: str) -> DocumentIssueTable | None:
        return (
            db.query(DocumentIssueTable)
            .filter(DocumentIssueTable.issue_id == issue_id)
            .one_or_none()
        )

    def save_issues(self, db: Session, issues: list[DocumentIssue]) -> list[DocumentIssueTable]:
        records = []
        for issue in issues:
            location = issue.location
            records.append(
                DocumentIssueTable(
                    issue_id=issue.issue_id,
                    document_id=issue.document_id,
                    review_id=issue.review_id,
                    issue_type=issue.issue_type,
                    severity=issue.severity,
                    original_text=issue.original_text,
                    problem_text=issue.problem_text,
                    suggestion=issue.suggestion,
                    reason=issue.reason,
                    confidence=issue.confidence,
                    page_number=location.page,
                    section_path=json.dumps(location.section_path, ensure_ascii=False),
                    paragraph_index=location.paragraph_index,
                    sentence_index=location.sentence_index,
                    start_offset=location.start_offset,
                    end_offset=location.end_offset,
                    status=issue.status,
                )
            )
        db.add_all(records)
        db.commit()
        return records

    def list_issues(
        self,
        db: Session,
        review_id: str,
        issue_type: str | None = None,
        min_confidence: float = 0.0,
        page: int | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[int, list[DocumentIssueTable]]:
        query = db.query(DocumentIssueTable).filter(
            DocumentIssueTable.review_id == review_id,
            DocumentIssueTable.confidence >= min_confidence,
        )
        if issue_type:
            query = query.filter(DocumentIssueTable.issue_type == issue_type)
        if page is not None:
            query = query.filter(DocumentIssueTable.page_number == page)
        if status:
            query = query.filter(DocumentIssueTable.status == status)
        total = query.count()
        items = query.order_by(DocumentIssueTable.id).offset(offset).limit(limit).all()
        return total, items

    def update_issue(
        self,
        db: Session,
        issue_id: str,
        status: str,
        comment: str | None,
    ) -> DocumentIssueTable | None:
        issue = self.get_issue(db, issue_id)
        if issue:
            issue.status = status
            issue.comment = comment
            db.commit()
            db.refresh(issue)
        return issue

    def detach_document(self, db: Session, document_id: str) -> None:
        (
            db.query(DocumentIssueTable)
            .filter(DocumentIssueTable.document_id == document_id)
            .delete(synchronize_session=False)
        )
        for review in db.query(ReviewJobTable).all():
            document_ids = json.loads(review.document_id or "[]")
            if document_id not in document_ids:
                continue
            remaining = [item for item in document_ids if item != document_id]
            if remaining:
                review.document_id = json.dumps(remaining, ensure_ascii=False)
            else:
                db.delete(review)
        db.flush()
