from typing import Any, Iterable, Optional


FINAL_RESPONSE_ARTIFACT_TYPE = "final_response"


def build_final_response_artifact(content: str) -> dict:
    return {
        "type": FINAL_RESPONSE_ARTIFACT_TYPE,
        "content": content or "",
    }


def extract_artifacts(payload: Any) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if hasattr(payload, "artifact"):
        artifact = getattr(payload, "artifact")
        if isinstance(artifact, list):
            return [item for item in artifact if isinstance(item, dict)]
    if isinstance(payload, (tuple, list)) and len(payload) == 2 and isinstance(payload[1], list):
        return [item for item in payload[1] if isinstance(item, dict)]
    return []


def find_final_response_artifact(artifacts: Iterable[dict]) -> Optional[str]:
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        if item.get("type") == FINAL_RESPONSE_ARTIFACT_TYPE:
            return str(item.get("content", "") or "")
    return None


def extract_final_response(payload: Any) -> Optional[str]:
    return find_final_response_artifact(extract_artifacts(payload))
