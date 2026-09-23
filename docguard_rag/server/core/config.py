import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "DocGuard-RAG"
    app_version: str = "0.6.0"
    environment: str = "development"
    debug: bool = False

    database_url: str = "sqlite:///./docguard.db"
    upload_dir: str = "uploads"
    max_upload_size: int = 20 * 1024 * 1024
    allowed_file_types: str = (
        "txt,md,markdown,html,htm,docx,pptx,pdf,png,jpg,jpeg,tif,tiff,bmp,webp"
    )
    ocr_lang: str = "chi_sim+eng"
    ocr_config: str = ""
    ocr_tesseract_cmd: str | None = None

    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_dimension: int = 128
    embedding_batch_size: int = 64
    embedding_query_cache_size: int = 1024

    vector_store_provider: str = "local"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "docguard_chunks"
    qdrant_timeout: float = 30.0

    embedding_provider: str = "hash"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model_name: str | None = None
    embedding_endpoint: str = "/embeddings"

    reranker_enabled: bool = True
    reranker_provider: str = "overlap"
    reranker_base_url: str | None = None
    reranker_api_key: str | None = None
    reranker_model_name: str | None = None
    reranker_endpoint: str = "/rerank"

    llm_provider: str = "extractive"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str | None = None
    llm_endpoint: str = "/chat/completions"
    llm_timeout: float = 60.0
    llm_temperature: float = 0.0
    llm_review_enabled: bool = False
    model_timeout: float = 30.0
    max_context_chars: int = 12000
    typo_confusion_json: str = "{}"

    corrective_rag_enabled: bool = True
    corrective_use_llm: bool = True
    corrective_max_retries: int = 1
    corrective_candidate_multiplier: int = 2
    corrective_min_relevance: float = 0.12
    corrective_min_evidence: float = 0.12
    corrective_min_verification: float = 0.6
    corrective_abstention_message: str = "文档中未找到足够且可验证的依据。"

    auth_enabled: bool = False
    api_keys_json: str = "{}"
    default_tenant_id: str = "default"
    support_history_limit: int = 20
    task_workers: int = 2
    task_backend: str = "local"
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"
    celery_task_max_retries: int = 3
    celery_task_retry_delay: int = 5

    @property
    def api_keys(self) -> dict[str, dict]:
        try:
            raw = json.loads(self.api_keys_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("DOCGUARD_API_KEYS_JSON must be valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("DOCGUARD_API_KEYS_JSON must be a JSON object")
        result = {}
        for key, value in raw.items():
            if isinstance(value, str):
                result[key] = {"tenant_id": value, "owner_id": None}
            elif isinstance(value, dict) and value.get("tenant_id"):
                result[key] = value
        return result

    @property
    def allowed_extensions(self) -> set[str]:
        """Return normalized configured upload extensions."""

        return {
            extension.strip().lower().lstrip(".")
            for extension in self.allowed_file_types.split(",")
            if extension.strip()
        }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOCGUARD_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""

    return Settings()
