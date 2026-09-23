from types import SimpleNamespace

from docguard_rag import DocGuard
from docguard_rag.server.parsers.image_parser import ImageOCRParser
from docguard_rag.server.schemas.customer_service import CustomerServiceRequest
from docguard_rag.server.services.customer_service import CustomerServiceService


class _RecordingQueryService:
    """Records the retrieval query it was handed instead of actually retrieving.

    These cases are about which turns reach the query builder, so a real corpus would only
    add fixture noise without changing what is being asserted.
    """

    def __init__(self, answer: str = "这是知识库里的答案。"):
        self.queries: list[str] = []
        self.answer = answer

    def query(self, db, payload, *, tenant_id, request_id=None):
        self.queries.append(payload.query)
        return SimpleNamespace(
            status="ANSWERED",
            answer=self.answer,
            sources=[],
            verified=True,
        )


def _guard(tmp_path, tenant_id: str = "default") -> DocGuard:
    return DocGuard(
        database_url=f"sqlite:///{tmp_path / 'history.db'}",
        upload_dir=tmp_path / "uploads",
        tenant_id=tenant_id,
    )


def _say(service, guard, message, *, session_id, tenant_id="default"):
    with guard._session_factory() as db:
        return service.chat(
            db,
            CustomerServiceRequest(message=message, session_id=session_id),
            tenant_id=tenant_id,
        )


def test_image_parser_advertises_common_raster_extensions():
    parser = ImageOCRParser()
    for extension in ("png", "jpg", "jpeg", "tiff", "webp"):
        assert parser.supports(extension)
    assert not parser.supports("pdf")


def test_local_customer_service_answers_and_handoffs(tmp_path):
    source = tmp_path / "service.md"
    source.write_text(
        "# 通境三创服务平台\n平台支持项目申报、材料提交和进度查询。\n",
        encoding="utf-8",
    )
    second_source = tmp_path / "faq.txt"
    second_source.write_text("账号注册需要手机号。", encoding="utf-8")

    with DocGuard(
        database_url=f"sqlite:///{tmp_path / 'customer.db'}",
        upload_dir=tmp_path / "uploads",
    ) as guard:
        documents = guard.documents.upload_many([source, second_source])
        assert len(documents) == 2
        document = documents[0]
        response = guard.customer_service.chat(
            "平台支持哪些服务？",
            document_ids=[document.document_id],
        )
        assert response.intent == "platform"
        assert response.status == "ANSWERED"
        assert response.sources
        assert "项目申报" in response.message

        handoff = guard.customer_service.chat("我想投诉并转人工", session_id=response.session_id)
        assert handoff.status == "HANDOFF_REQUIRED"
        assert handoff.handoff_required is True

        guard.customer_service.reset(response.session_id)


def test_history_survives_a_fresh_service_instance(tmp_path):
    """The whole point of persisting: a new instance, or a new process, sees old turns."""

    with _guard(tmp_path) as guard:
        first = _say(
            CustomerServiceService(_RecordingQueryService()),
            guard,
            "平台支持哪些服务？",
            session_id="cs_shared",
        )

        reborn = CustomerServiceService(_RecordingQueryService())
        with guard._session_factory() as db:
            history = reborn.sessions.load_history(
                db,
                tenant_id="default",
                session_id="cs_shared",
                limit=reborn.history_limit,
            )

    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["content"] == "平台支持哪些服务？"
    assert history[1]["content"] == first.message


def test_the_contextual_query_sees_only_previous_turns(tmp_path):
    recorder = _RecordingQueryService()
    service = CustomerServiceService(recorder)

    with _guard(tmp_path) as guard:
        _say(service, guard, "第一个问题", session_id="cs_ctx")
        _say(service, guard, "第二个问题", session_id="cs_ctx")
        _say(service, guard, "第三个问题", session_id="cs_ctx")

    # Turn one has no history, so it is passed through untouched.
    assert recorder.queries[0] == "第一个问题"
    # Later turns see the earlier user messages...
    assert "第一个问题" in recorder.queries[1]
    assert "第二个问题" in recorder.queries[1]
    # ...but never the one being answered, which only appears after the marker.
    previous, current = recorder.queries[2].split("\n当前问题：")
    assert previous == "此前对话：第一个问题；第二个问题"
    assert current == "第三个问题"


def test_history_is_trimmed_exactly_to_the_limit(tmp_path):
    service = CustomerServiceService(_RecordingQueryService(), history_limit=4)

    with _guard(tmp_path) as guard:
        for index in range(6):
            _say(service, guard, f"问题 {index}", session_id="cs_trim")
        with guard._session_factory() as db:
            stored = service.sessions.load_history(
                db,
                tenant_id="default",
                session_id="cs_trim",
                limit=100,
            )
            session = service.sessions.get_or_create(
                db,
                tenant_id="default",
                session_id="cs_trim",
            )
            count = session.message_count

    # Matches deque(maxlen=4): the oldest turns are dropped, not kept and re-read.
    assert [item["content"] for item in stored] == [
        "问题 4",
        "这是知识库里的答案。",
        "问题 5",
        "这是知识库里的答案。",
    ]
    assert count == 4


def test_reset_clears_history_and_reports_unknown_sessions(tmp_path):
    service = CustomerServiceService(_RecordingQueryService())

    with _guard(tmp_path) as guard:
        _say(service, guard, "平台支持哪些服务？", session_id="cs_reset")
        with guard._session_factory() as db:
            assert service.reset(db, "cs_reset", tenant_id="default") is True
        with guard._session_factory() as db:
            assert service.reset(db, "cs_never_existed", tenant_id="default") is False
        with guard._session_factory() as db:
            stored = service.sessions.load_history(
                db,
                tenant_id="default",
                session_id="cs_reset",
                limit=100,
            )

    assert stored == []


def test_a_reset_session_id_can_be_reused(tmp_path):
    recorder = _RecordingQueryService()
    service = CustomerServiceService(recorder)

    with _guard(tmp_path) as guard:
        _say(service, guard, "第一个问题", session_id="cs_reuse")
        with guard._session_factory() as db:
            service.reset(db, "cs_reuse", tenant_id="default")
        _say(service, guard, "重新开始", session_id="cs_reuse")

    # The reused id starts from an empty history, which is what a client expects.
    assert recorder.queries[-1] == "重新开始"


def test_two_tenants_do_not_share_a_session_id(tmp_path):
    alpha_service = CustomerServiceService(_RecordingQueryService())
    beta_service = CustomerServiceService(_RecordingQueryService(answer="另一个租户的答案。"))

    with _guard(tmp_path, tenant_id="alpha") as alpha, _guard(tmp_path, tenant_id="beta") as beta:
        _say(alpha_service, alpha, "alpha 的问题", session_id="cs_same_id", tenant_id="alpha")
        _say(beta_service, beta, "beta 的问题", session_id="cs_same_id", tenant_id="beta")

        with alpha._session_factory() as db:
            alpha_history = alpha_service.sessions.load_history(
                db,
                tenant_id="alpha",
                session_id="cs_same_id",
                limit=100,
            )
            beta_history = alpha_service.sessions.load_history(
                db,
                tenant_id="beta",
                session_id="cs_same_id",
                limit=100,
            )
            # alpha cannot reset beta's conversation either.
            assert alpha_service.reset(db, "cs_same_id", tenant_id="alpha") is True
            beta_after_reset = beta_service.sessions.load_history(
                db,
                tenant_id="beta",
                session_id="cs_same_id",
                limit=100,
            )

    assert [item["content"] for item in alpha_history] == ["alpha 的问题", "这是知识库里的答案。"]
    assert [item["content"] for item in beta_history] == ["beta 的问题", "另一个租户的答案。"]
    assert [item["content"] for item in beta_after_reset] == ["beta 的问题", "另一个租户的答案。"]
