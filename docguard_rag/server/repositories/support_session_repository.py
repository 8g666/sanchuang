"""Persistence for /v1/support/chat conversation history.

Replaces the process-local defaultdict of deques that CustomerServiceService used to hold.
That structure lost every conversation on restart, was invisible to other workers, and grew
without bound by session count. Every read and write here is scoped by tenant_id together
with session_id, because callers may supply their own session_id and two tenants can pick
the same one.
"""

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from docguard_rag.server.database.models import SupportMessageTable, SupportSessionTable


class SupportSessionRepository:
    def get_or_create(self, db: Session, *, tenant_id: str, session_id: str) -> SupportSessionTable:
        record = self._get(db, tenant_id=tenant_id, session_id=session_id)
        if record is not None:
            if record.status != "ACTIVE":
                # A reset session id is reusable: the caller expects an empty history, not
                # a permanently dead session.
                record.status = "ACTIVE"
                db.commit()
            return record

        record = SupportSessionTable(
            session_id=session_id,
            tenant_id=tenant_id,
            status="ACTIVE",
            message_count=0,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def load_history(
        self,
        db: Session,
        *,
        tenant_id: str,
        session_id: str,
        limit: int,
    ) -> list[dict[str, str]]:
        """The stored turns, oldest first, in the shape the query builder already expects."""

        rows = db.execute(
            select(SupportMessageTable.role, SupportMessageTable.content)
            .where(
                SupportMessageTable.tenant_id == tenant_id,
                SupportMessageTable.session_id == session_id,
            )
            .order_by(SupportMessageTable.id.desc())
            .limit(limit)
        ).all()
        # Ordered by id, not created_at: a turn's two rows share a request and created_at
        # has only millisecond precision, so it would tie and leave the pair unordered.
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def append_turn(
        self,
        db: Session,
        *,
        tenant_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        history_limit: int,
        intent: str | None = None,
        status: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Store one turn and trim to the newest `history_limit` messages.

        Trimmed on write rather than read: storing everything and reading the last N would
        make the table grow forever for a feature that only ever looks at N rows.
        """

        now = datetime.now()
        db.add(
            SupportMessageTable(
                session_id=session_id,
                tenant_id=tenant_id,
                role="user",
                content=user_message,
                intent=intent,
                status=status,
                request_id=request_id,
                created_at=now,
            )
        )
        db.add(
            SupportMessageTable(
                session_id=session_id,
                tenant_id=tenant_id,
                role="assistant",
                content=assistant_message,
                intent=intent,
                status=status,
                request_id=request_id,
                created_at=now,
            )
        )

        session = self._get(db, tenant_id=tenant_id, session_id=session_id)
        if session is not None:
            session.message_count = min(session.message_count + 2, history_limit)
            session.last_message_at = now
            session.status = "ACTIVE"

        # SessionLocal is autoflush=False, so without this the trim would compute its
        # cutoff over the rows that existed before this turn and keep limit + 2.
        db.flush()
        self._trim(db, tenant_id=tenant_id, session_id=session_id, limit=history_limit)
        # get_db only yields and closes, so anything that must persist commits for itself.
        db.commit()

    def reset(self, db: Session, *, tenant_id: str, session_id: str) -> bool:
        """Soft reset: drop the turns, keep the session row. Returns whether it existed.

        Keeping the row leaves an audit trail that the conversation happened and was
        reset, and lets the response distinguish "reset" from "never existed" without
        turning a fire-and-forget DELETE into a 404 for callers.
        """

        session = self._get(db, tenant_id=tenant_id, session_id=session_id)
        db.query(SupportMessageTable).filter(
            SupportMessageTable.tenant_id == tenant_id,
            SupportMessageTable.session_id == session_id,
        ).delete(synchronize_session=False)
        if session is not None:
            session.status = "RESET"
            session.message_count = 0
        db.commit()
        return session is not None

    @staticmethod
    def _get(db: Session, *, tenant_id: str, session_id: str) -> SupportSessionTable | None:
        return (
            db.query(SupportSessionTable)
            .filter(
                SupportSessionTable.tenant_id == tenant_id,
                SupportSessionTable.session_id == session_id,
            )
            .one_or_none()
        )

    @staticmethod
    def _trim(db: Session, *, tenant_id: str, session_id: str, limit: int) -> None:
        scoped = (
            SupportMessageTable.tenant_id == tenant_id,
            SupportMessageTable.session_id == session_id,
        )
        newest = (
            select(SupportMessageTable.id)
            .where(*scoped)
            .order_by(SupportMessageTable.id.desc())
            .limit(limit)
            .subquery()
        )
        cutoff = select(func.min(newest.c.id)).scalar_subquery()
        db.query(SupportMessageTable).filter(*scoped, SupportMessageTable.id < cutoff).delete(
            synchronize_session=False
        )
