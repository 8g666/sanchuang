import json
from uuid import uuid4

from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.issue import DocumentIssue, IssueLocation
from docguard_rag.server.reviewers.base import DocumentReviewer


class LLMReviewer(DocumentReviewer):
    """General language reviewer using a configured structured-output LLM."""

    issue_types = {
        "TYPO",
        "GRAMMAR",
        "SEMANTIC_UNCLEAR",
        "CONTEXT_BREAK",
        "TERM_INCONSISTENT",
        "FACT_CONFLICT",
    }

    def __init__(self, provider):
        self.provider = provider

    def review(
        self,
        block: DocumentBlock,
        review_id: str,
        requested_type: str,
    ) -> list[DocumentIssue]:
        response = self.provider.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "你是高精度文档校对器。只报告明确问题，不把写作风格偏好当作错误。"
                        "建议不得改变原意或事实。返回 JSON，字符位置相对于给定文本。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"检查类型：{requested_type}\n"
                        f"文本：{block.text}\n"
                        "返回 issues 数组；每项包含 problem_text、suggestion、reason、"
                        "confidence、severity、start_offset、end_offset。"
                    ),
                },
            ],
            response_schema={
                "name": "document_review",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "problem_text": {"type": "string"},
                                    "suggestion": {"type": "string"},
                                    "reason": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "severity": {"type": "string"},
                                    "start_offset": {"type": "integer"},
                                    "end_offset": {"type": "integer"},
                                },
                                "required": [
                                    "problem_text",
                                    "suggestion",
                                    "reason",
                                    "confidence",
                                    "severity",
                                    "start_offset",
                                    "end_offset",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["issues"],
                    "additionalProperties": False,
                },
            },
        )
        content = response.get("content", "{}")
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            try:
                body = json.loads(cleaned or "{}")
            except json.JSONDecodeError as exc:
                raise ProviderError("LLM review provider returned invalid JSON") from exc
        else:
            body = content

        if not isinstance(body, dict) or not isinstance(body.get("issues", []), list):
            raise ProviderError("LLM review provider returned an invalid result shape")

        issues = []
        for item in body.get("issues", [])[:20]:
            problem = str(item.get("problem_text", ""))
            suggestion = str(item.get("suggestion", ""))
            if not problem or not suggestion or problem == suggestion:
                continue
            start = int(item.get("start_offset", -1))
            end = int(item.get("end_offset", -1))
            if start < 0 or end > len(block.text) or block.text[start:end] != problem:
                start = block.text.find(problem)
                end = start + len(problem)
            if start < 0:
                continue
            absolute_start = (block.start_offset or 0) + start
            issues.append(
                DocumentIssue(
                    issue_id=f"issue_{uuid4().hex[:12]}",
                    document_id=block.document_id,
                    review_id=review_id,
                    issue_type=requested_type,
                    severity=str(item.get("severity", "MEDIUM")).upper(),
                    original_text=block.text,
                    problem_text=problem,
                    suggestion=suggestion,
                    reason=str(item.get("reason", "模型复核发现明确表达问题。")),
                    confidence=max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
                    location=IssueLocation(
                        page=block.page_number,
                        section_path=list(block.section_path),
                        paragraph_index=block.paragraph_index,
                        sentence_index=block.sentence_index,
                        start_offset=absolute_start,
                        end_offset=absolute_start + len(problem),
                    ),
                )
            )
        return issues
