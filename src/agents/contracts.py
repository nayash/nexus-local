import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


WorkerName = Literal[
    "local_retrieval_worker",
    "local_catalog_worker",
    "web_retrieval_worker",
    "identity_worker",
    "tabular_worker",
    "none",
]


class IntentPacket(BaseModel):
    primary_intent: Literal[
        "local_content",
        "local_metadata",
        "local_catalog",
        "web",
        "identity",
        "tabular",
        "hybrid_local_web",
        "general_chat",
    ] = "general_chat"
    normalized_query: str = ""
    requires_hybrid_local: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    rationale: str = ""


class WorkerTask(BaseModel):
    worker: WorkerName = "none"
    objective: str = ""
    query: str = ""
    mode: str = ""
    required_evidence: List[str] = Field(default_factory=list)


class ManagerDecision(BaseModel):
    action: Literal["dispatch", "synthesize"] = "synthesize"
    continue_reasoning: bool = False
    next_task: WorkerTask = Field(default_factory=WorkerTask)
    reason: str = ""


class EvidenceItem(BaseModel):
    source_type: Literal["local", "web", "system"] = "system"
    title: str = ""
    url: str = ""
    snippet: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkerResult(BaseModel):
    worker: WorkerName = "none"
    status: Literal["ok", "empty", "error"] = "empty"
    summary: str = ""
    proposed_answer: str = ""
    evidence: List[EvidenceItem] = Field(default_factory=list)
    source_metadata: List[Dict[str, Any]] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    items: List[EvidenceItem] = Field(default_factory=list)
    source_metadata: List[Dict[str, Any]] = Field(default_factory=list)
    worker_summaries: List[str] = Field(default_factory=list)


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    candidate = (raw_text or "").strip()
    if not candidate:
        return {}

    if "{" in candidate and "}" in candidate:
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        if end > start:
            candidate = candidate[start:end]

    try:
        payload = json.loads(candidate)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
