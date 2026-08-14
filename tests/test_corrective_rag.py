from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docguard_rag.server.database.connection import Base
from docguard_rag.server.schemas.query import QueryRequest, QuerySource
from docguard_rag.server.services.answer_service import AnswerService
from docguard_rag.server.services.corrective_rag_service import (
    AnswerVerifier,
    QueryRewriter,
    RetrievalGrader,
)
from docguard_rag.server.services.query_service import QueryService


class SequencedRetrieval:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def search(self, query, top_k, document_ids=None, filters=None):
        self.calls.append((query, top_k))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class UnsupportedAnswerService:
    provider = None

    def generate(self, query, sources):
        return "Mars is red."


def raw_source(chunk_id: str, text: str, score: float = 1.0):
    return {
        "chunk_id": chunk_id,
        "score": score,
        "metadata": {"document_id": "doc-test", "text": text},
    }


def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'corrective.db'}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)()


def test_query_rewriter_corrects_typo_without_model():
    rewriter = QueryRewriter(provider=None, use_llm=False)

    result = rewriter.rewrite("如何掉用接口")

    assert result.query == "如何调用接口"
    assert result.changed is True


def test_corrective_query_retries_then_returns_verified_answer(tmp_path):
    retrieval = SequencedRetrieval(
        [
            [raw_source("irrelevant", "Weather forecast for tomorrow.")],
            [raw_source("relevant", "The API call uses the /v1 endpoint.")],
        ]
    )
    service = QueryService(
        retrieval_service=retrieval,
        answer_service=AnswerService(provider=None),
        query_rewriter=QueryRewriter(provider=None, use_llm=False),
        retrieval_grader=RetrievalGrader(provider=None, use_llm=False),
        answer_verifier=AnswerVerifier(provider=None, use_llm=False),
    )
    engine, db = session(tmp_path)
    try:
        result = service.query(
            db,
            QueryRequest(query="How do I call API?", top_k=2, max_retries=1),
        )
    finally:
        db.close()
        engine.dispose()

    assert result.status == "ANSWERED"
    assert result.verified is True
    assert result.attempt_count == 2
    assert result.attempts[0].evidence_sufficient is False
    assert result.attempts[1].evidence_sufficient is True
    assert result.sources[0].chunk_id == "relevant"
    assert retrieval.calls[1][1] > retrieval.calls[0][1]


def test_corrective_query_abstains_when_evidence_stays_insufficient(tmp_path):
    retrieval = SequencedRetrieval(
        [[raw_source("irrelevant", "API authentication token configuration.")]]
    )
    service = QueryService(
        retrieval_service=retrieval,
        answer_service=AnswerService(provider=None),
        query_rewriter=QueryRewriter(provider=None, use_llm=False),
        retrieval_grader=RetrievalGrader(provider=None, use_llm=False),
        answer_verifier=AnswerVerifier(provider=None, use_llm=False),
    )
    engine, db = session(tmp_path)
    try:
        result = service.query(
            db,
            QueryRequest(query="moon distance", max_retries=1),
        )
    finally:
        db.close()
        engine.dispose()

    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.verified is False
    assert result.attempt_count == 2
    assert "依据" in result.answer
    assert all(not attempt.evidence_sufficient for attempt in result.attempts)


def test_corrective_query_rejects_an_unsupported_answer(tmp_path):
    retrieval = SequencedRetrieval([[raw_source("api", "The API call uses the /v1 endpoint.")]])
    service = QueryService(
        retrieval_service=retrieval,
        answer_service=UnsupportedAnswerService(),
        query_rewriter=QueryRewriter(provider=None, use_llm=False),
        retrieval_grader=RetrievalGrader(provider=None, use_llm=False),
        answer_verifier=AnswerVerifier(provider=None, use_llm=False),
    )
    engine, db = session(tmp_path)
    try:
        result = service.query(
            db,
            QueryRequest(query="Which API endpoint is used?", max_retries=0),
        )
    finally:
        db.close()
        engine.dispose()

    assert result.status == "ANSWER_UNVERIFIED"
    assert result.verified is False
    assert "依据" in result.answer
    assert result.attempts[0].answer_verified is False


def test_corrective_can_be_disabled_for_legacy_single_pass(tmp_path):
    retrieval = SequencedRetrieval(
        [[raw_source("irrelevant", "API authentication token configuration.")]]
    )
    service = QueryService(
        retrieval_service=retrieval,
        answer_service=AnswerService(provider=None),
    )
    engine, db = session(tmp_path)
    try:
        result = service.query(
            db,
            QueryRequest(query="moon distance", corrective=False),
        )
    finally:
        db.close()
        engine.dispose()

    assert result.status == "ANSWERED"
    assert result.answer == "API authentication token configuration."
    assert result.attempt_count == 1


def test_model_grader_and_verifier_support_semantic_evidence():
    class Provider:
        def generate(self, messages, response_schema=None):
            name = response_schema["name"]
            if name == "retrieval_grades":
                return {"content": {"items": [{"chunk_id": "semantic", "relevance": 0.94}]}}
            return {
                "content": {
                    "supported": True,
                    "confidence": 0.91,
                    "reason": "all claims are supported",
                }
            }

    provider = Provider()
    source = QuerySource(
        chunk_id="semantic",
        document_id="doc",
        text="The service accepts requests at /v1.",
        score=0.2,
    )

    assessment = RetrievalGrader(provider=provider, use_llm=True).grade(
        "Where should clients send requests?",
        [source],
    )
    verification = AnswerVerifier(provider=provider, use_llm=True).verify(
        "Where should clients send requests?",
        "Clients should use /v1. [S1]",
        [source],
    )
    invalid_citation = AnswerVerifier(provider=provider, use_llm=True).verify(
        "Where should clients send requests?",
        "Clients should use /v1. [S2]",
        [source],
    )

    assert assessment.sufficient is True
    assert assessment.relevant_sources[0].relevance_score == 0.94
    assert verification.verified is True
    assert invalid_citation.verified is False
