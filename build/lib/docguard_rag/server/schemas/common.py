from typing import Generic, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    """Unified API response structure."""

    code: int = Field(default=0, description="Business status code")
    message: str = Field(default="success", description="Response message")
    request_id: str = Field(description="Unique request identifier")
    data: DataT | None = Field(default=None, description="Response data")
