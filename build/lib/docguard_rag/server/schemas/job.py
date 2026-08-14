from pydantic import BaseModel


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_stage: str | None = None
    error: str | None = None
