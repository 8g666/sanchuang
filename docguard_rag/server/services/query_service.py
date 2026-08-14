from time import perf_counter
from uuid import uuid4

from sqlalchemy.orm import Session

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.repositories.query_log_repository import QueryLogRepository
from docguard_rag.server.schemas.query import (
    QueryAttempt,
    QueryRequest,
    QueryResult,
    QuerySource,
)
from docguard_rag.server.services.answer_service import AnswerService
from docguard_rag.server.services.corrective_rag_service import (
    AnswerVerifier,
    EvidenceAssessment,
    QueryRewriter,
    RetrievalGrader,
)
from docguard_rag.server.services.retrieval_service import RetrievalService


class QueryService:
    """Run evidence-first retrieval with an optional corrective feedback loop."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        answer_service: AnswerService | None = None,
        query_rewriter: QueryRewriter | None = None,
        retrieval_grader: RetrievalGrader | None = None,
        answer_verifier: AnswerVerifier | None = None,
    ):
        self.retrieval_service = retrieval_service or RetrievalService()
        self.answer_service = answer_service or AnswerService()
        provider = getattr(self.answer_service, "provider", None)
        self.query_rewriter = query_rewriter or QueryRewriter(provider=provider)
        self.retrieval_grader = retrieval_grader or RetrievalGrader(provider=provider)
        self.answer_verifier = answer_verifier or AnswerVerifier(provider=provider)
        self.document_repository = DocumentRepository()
        self.query_log_repository = QueryLogRepository()

    def query(
        self,
        db: Session,
        payload: QueryRequest,
        tenant_id: str = "default",
        request_id: str | None = None,
    ) -> QueryResult:
        started_at = perf_counter()
        request_id = request_id or f"qry_{uuid4().hex}"
        settings = get_settings()
        corrective = (
            settings.corrective_rag_enabled if payload.corrective is None else payload.corrective
        )
        filters = dict(payload.filters)
        filters["tenant_id"] = tenant_id
        if payload.collection_id:
            filters["collection_id"] = payload.collection_id

        if corrective:
            result = self._corrective_query(db, payload, filters, tenant_id)
        else:
            result = self._single_pass_query(db, payload, filters, tenant_id)

        self.query_log_repository.create(
            db,
            request_id=request_id,
            tenant_id=tenant_id,
            query_text=payload.query,
            document_ids=payload.document_ids,
            retrieved_chunk_ids=[source.chunk_id for source in result.sources],
            answer_text=result.answer,
            confidence=result.confidence,
            latency_ms=max(0, round((perf_counter() - started_at) * 1000)),
            token_usage={
                "corrective_rag": {
                    "status": result.status,
                    "effective_query": result.effective_query,
                    "query_rewritten": result.query_rewritten,
                    "verified": result.verified,
                    "attempts": [attempt.model_dump() for attempt in result.attempts],
                }
            },
        )
        return result

    def _corrective_query(
        self,
        db: Session,
        payload: QueryRequest,
        filters: dict,
        tenant_id: str,
    ) -> QueryResult:
        settings = get_settings()
        max_retries = (
            settings.corrective_max_retries if payload.max_retries is None else payload.max_retries
        )
        max_retries = max(0, min(3, max_retries))
        multiplier = max(1, settings.corrective_candidate_multiplier)
        rewrite = self.query_rewriter.rewrite(payload.query, attempt=0)
        current_query = rewrite.query
        attempts: list[QueryAttempt] = []
        best_assessment: EvidenceAssessment | None = None
        best_query = current_query
        verification_failed = False

        for attempt_index in range(max_retries + 1):
            candidate_count = min(
                50,
                max(payload.top_k, payload.top_k * (multiplier**attempt_index)),
            )
            raw_results = self.retrieval_service.search(
                current_query,
                candidate_count,
                document_ids=payload.document_ids,
                filters=filters,
            )
            sources = self._to_sources(db, raw_results, tenant_id)
            assessment = self.retrieval_grader.grade(current_query, sources)
            if best_assessment is None or assessment.score > best_assessment.score:
                best_assessment = assessment
                best_query = current_query

            attempt = QueryAttempt(
                attempt=attempt_index + 1,
                query=current_query,
                retrieved_count=len(sources),
                relevant_count=len(assessment.relevant_sources),
                evidence_score=assessment.score,
                evidence_sufficient=assessment.sufficient,
                reason=f"{rewrite.reason}; {assessment.reason}",
            )

            if assessment.sufficient:
                selected = assessment.relevant_sources[: payload.top_k]
                if not payload.generate_answer:
                    attempts.append(attempt)
                    return self._result(
                        payload,
                        answer=None,
                        confidence=assessment.score,
                        sources=selected,
                        status="RETRIEVAL_ONLY",
                        effective_query=current_query,
                        verified=None,
                        attempts=attempts,
                    )

                answer = self.answer_service.generate(current_query, selected)
                verification = self.answer_verifier.verify(
                    current_query,
                    answer,
                    selected,
                )
                attempt.answer_verified = verification.verified
                attempt.verification_score = verification.score
                attempt.reason = f"{attempt.reason}; {verification.reason}"
                attempts.append(attempt)
                if verification.verified:
                    return self._result(
                        payload,
                        answer=answer,
                        confidence=min(assessment.score, verification.score),
                        sources=selected,
                        status="ANSWERED",
                        effective_query=current_query,
                        verified=True,
                        attempts=attempts,
                    )
                verification_failed = True
                feedback = "the generated answer was not fully supported: " + verification.reason
            else:
                attempts.append(attempt)
                feedback = "retrieved evidence was insufficient: " + assessment.reason

            if attempt_index < max_retries:
                rewrite = self.query_rewriter.rewrite(
                    payload.query,
                    current_query,
                    attempt=attempt_index + 1,
                    feedback=feedback,
                )
                current_query = rewrite.query

        fallback = best_assessment or EvidenceAssessment([], [], 0.0, False, "no evidence")
        fallback_sources = (
            fallback.relevant_sources[: payload.top_k]
            if fallback.relevant_sources
            else fallback.sources[: payload.top_k]
        )
        status = "ANSWER_UNVERIFIED" if verification_failed else "INSUFFICIENT_EVIDENCE"
        return self._result(
            payload,
            answer=(settings.corrective_abstention_message if payload.generate_answer else None),
            confidence=fallback.score,
            sources=fallback_sources,
            status=status,
            effective_query=best_query,
            verified=False if payload.generate_answer else None,
            attempts=attempts,
        )

    def _single_pass_query(
        self,
        db: Session,
        payload: QueryRequest,
        filters: dict,
        tenant_id: str,
    ) -> QueryResult:
        raw_results = self.retrieval_service.search(
            payload.query,
            payload.top_k,
            document_ids=payload.document_ids,
            filters=filters,
        )
        sources = self._to_sources(db, raw_results, tenant_id)
        answer = (
            self.answer_service.generate(payload.query, sources)
            if payload.generate_answer
            else None
        )
        confidence = round(
            sum(source.score for source in sources[:3]) / max(1, min(3, len(sources))),
            6,
        )
        return self._result(
            payload,
            answer=answer,
            confidence=confidence,
            sources=sources,
            status="ANSWERED" if payload.generate_answer else "RETRIEVAL_ONLY",
            effective_query=payload.query,
            verified=None,
            attempts=[
                QueryAttempt(
                    attempt=1,
                    query=payload.query,
                    retrieved_count=len(sources),
                    relevant_count=len(sources),
                    evidence_score=confidence,
                    evidence_sufficient=bool(sources),
                    reason="corrective RAG disabled",
                )
            ],
        )

    def _to_sources(
        self,
        db: Session,
        results: list[dict],
        tenant_id: str,
    ) -> list[QuerySource]:
        document_ids = list(
            {
                item.get("metadata", {}).get("document_id")
                for item in results
                if item.get("metadata", {}).get("document_id")
            }
        )
        filenames = {
            document.document_id: document.filename
            for document in self.document_repository.get_documents(
                db,
                document_ids,
                tenant_id=tenant_id,
            )
        }
        return [
            QuerySource(
                chunk_id=item["chunk_id"],
                document_id=item.get("metadata", {}).get("document_id", ""),
                filename=filenames.get(item.get("metadata", {}).get("document_id")),
                page=item.get("metadata", {}).get("page"),
                section_path=item.get("metadata", {}).get("section_path", []),
                paragraph_index=item.get("metadata", {}).get("paragraph_index"),
                start_offset=item.get("metadata", {}).get("start_offset"),
                end_offset=item.get("metadata", {}).get("end_offset"),
                text=item.get("metadata", {}).get("text", ""),
                score=float(item.get("score", 0.0)),
            )
            for item in results
        ]

    @staticmethod
    def _result(
        payload: QueryRequest,
        *,
        answer: str | None,
        confidence: float,
        sources: list[QuerySource],
        status: str,
        effective_query: str,
        verified: bool | None,
        attempts: list[QueryAttempt],
    ) -> QueryResult:
        return QueryResult(
            answer=answer,
            confidence=round(confidence, 6),
            sources=sources if payload.include_sources else [],
            status=status,
            original_query=payload.query,
            effective_query=effective_query,
            query_rewritten=effective_query != payload.query.strip(),
            attempt_count=len(attempts),
            verified=verified,
            attempts=attempts,
        )
