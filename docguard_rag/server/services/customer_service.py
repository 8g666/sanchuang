from collections.abc import Sequence
from uuid import uuid4

from sqlalchemy.orm import Session

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.repositories.support_session_repository import SupportSessionRepository
from docguard_rag.server.schemas.customer_service import (
    CustomerServiceRequest,
    CustomerServiceResponse,
)
from docguard_rag.server.schemas.query import QueryRequest
from docguard_rag.server.services.query_service import QueryService


class CustomerServiceService:
    """Small, provider-neutral AI customer service built on the RAG query loop."""

    _intent_terms = {
        "human": ("人工", "客服", "投诉", "举报", "转人工", "真人", "operator"),
        "application": ("申报", "报名", "申请", "项目", "材料", "截止", "流程"),
        "account": ("账号", "登录", "密码", "注册", "绑定", "权限"),
        "platform": ("平台", "系统", "功能", "怎么用", "使用", "帮助", "咨询"),
    }

    def __init__(
        self,
        query_service: QueryService | None = None,
        history_limit: int | None = None,
        sessions: SupportSessionRepository | None = None,
    ):
        self.query_service = query_service or QueryService()
        self.history_limit = history_limit or get_settings().support_history_limit
        self.sessions = sessions or SupportSessionRepository()

    def chat(
        self,
        db: Session,
        payload: CustomerServiceRequest,
        *,
        tenant_id: str = "default",
        request_id: str | None = None,
    ) -> CustomerServiceResponse:
        session_id = payload.session_id or f"cs_{uuid4().hex[:16]}"
        intent, intent_confidence = self.classify_intent(payload.message)
        self.sessions.get_or_create(db, tenant_id=tenant_id, session_id=session_id)
        history = self.sessions.load_history(
            db,
            tenant_id=tenant_id,
            session_id=session_id,
            limit=self.history_limit,
        )

        if intent == "human":
            response = CustomerServiceResponse(
                session_id=session_id,
                message="我已记录你的诉求，正在为你转接人工客服。",
                intent=intent,
                intent_confidence=intent_confidence,
                status="HANDOFF_REQUIRED",
                handoff_required=True,
                suggested_actions=["请留下联系电话或工单号", "人工客服工作时间为工作日 9:00-18:00"],
            )
        else:
            query = QueryRequest(
                query=self._contextual_query(payload.message, history),
                document_ids=payload.document_ids,
                collection_id=payload.collection_id,
                top_k=payload.top_k,
                corrective=payload.corrective,
                max_retries=payload.max_retries,
            )
            result = self.query_service.query(
                db,
                query,
                tenant_id=tenant_id,
                request_id=request_id or f"cs_{uuid4().hex}",
            )
            unresolved = result.status in {"INSUFFICIENT_EVIDENCE", "ANSWER_UNVERIFIED"}
            response = CustomerServiceResponse(
                session_id=session_id,
                message=(
                    "目前知识库中没有足够依据回答这个问题，我可以为你转接人工客服。"
                    if unresolved
                    else (result.answer or "请提供更具体的问题。")
                ),
                intent=intent,
                intent_confidence=intent_confidence,
                status="HANDOFF_REQUIRED"
                if unresolved and payload.handoff_if_unresolved
                else result.status,
                handoff_required=unresolved and payload.handoff_if_unresolved,
                suggested_actions=self._suggestions(intent, unresolved),
                sources=result.sources,
                query_status=result.status,
                query_verified=result.verified,
            )

        # Appended after the response is built, and committed by the repository: the query
        # above logs and commits mid-request, and _contextual_query must never see the
        # message it is answering.
        self.sessions.append_turn(
            db,
            tenant_id=tenant_id,
            session_id=session_id,
            user_message=payload.message,
            assistant_message=response.message,
            history_limit=self.history_limit,
            intent=intent,
            status=response.status,
            request_id=request_id,
        )
        return response

    def reset(self, db: Session, session_id: str, *, tenant_id: str = "default") -> bool:
        """Clear a conversation's history. Returns whether the session existed."""

        return self.sessions.reset(db, tenant_id=tenant_id, session_id=session_id)

    @classmethod
    def classify_intent(cls, message: str) -> tuple[str, float]:
        normalized = message.lower().strip()
        scores = {
            intent: sum(term.lower() in normalized for term in terms)
            for intent, terms in cls._intent_terms.items()
        }
        intent, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            return "knowledge", 0.45
        return intent, round(min(0.98, 0.58 + score * 0.1), 4)

    @staticmethod
    def _contextual_query(message: str, history: Sequence[dict[str, str]]) -> str:
        previous = [item["content"] for item in history[-4:] if item["role"] == "user"]
        if not previous:
            return message
        return "此前对话：" + "；".join(previous) + "\n当前问题：" + message

    @staticmethod
    def _suggestions(intent: str, unresolved: bool) -> list[str]:
        if unresolved:
            return ["换一种方式描述问题", "查看平台办事指南", "转人工客服"]
        suggestions = {
            "application": ["查看申报材料清单", "查看项目申报流程"],
            "account": ["查看账号帮助", "转人工客服"],
            "platform": ["查看平台使用指南", "继续提问"],
            "knowledge": ["继续提问", "查看相关文档"],
        }
        return suggestions.get(intent, ["继续提问"])
