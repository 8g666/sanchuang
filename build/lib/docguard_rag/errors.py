class DocGuardError(RuntimeError):
    """Base exception raised by the public package."""


class DocGuardAPIError(DocGuardError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
