from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    timestamp: float
    request_timestamp: float | None = None
    method: str
    path: str
    request: dict
    response: dict
    status_code: int
    # Present only for the opt-in CompactionRL protocol.  Defaults keep old
    # session dumps readable and make an unmarked record unambiguously legacy.
    compaction_schema_version: int | None = None
    compaction_context_window: int | None = None
    compaction_segment_index: int | None = None
    compaction_segment_type: str | None = None
    compaction_context_budget: int | None = None


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
