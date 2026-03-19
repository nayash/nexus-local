import multiprocessing as mp
import queue
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.config import Config


def _serialize_message(message) -> dict[str, Any]:
    return {
        "type": getattr(message, "type", ""),
        "content": getattr(message, "content", "") or "",
    }


def _deserialize_message(payload: dict[str, Any]):
    message_type = str(payload.get("type", "") or "").strip().lower()
    content = payload.get("content", "") or ""
    if message_type == "system":
        return SystemMessage(content=content)
    if message_type == "ai":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _serialize_response(response) -> dict[str, Any]:
    return {
        "content": getattr(response, "content", "") or "",
        "additional_kwargs": getattr(response, "additional_kwargs", {}) or {},
        "response_metadata": getattr(response, "response_metadata", {}) or {},
        "tool_calls": getattr(response, "tool_calls", []) or [],
        "usage_metadata": getattr(response, "usage_metadata", None),
        "id": getattr(response, "id", None),
        "name": getattr(response, "name", None),
    }


def _build_ai_message(payload: dict[str, Any]) -> AIMessage:
    kwargs = {
        "content": payload.get("content", "") or "",
        "additional_kwargs": payload.get("additional_kwargs", {}) or {},
        "response_metadata": payload.get("response_metadata", {}) or {},
        "tool_calls": payload.get("tool_calls", []) or [],
    }
    usage_metadata = payload.get("usage_metadata")
    if usage_metadata is not None:
        kwargs["usage_metadata"] = usage_metadata
    message_id = payload.get("id")
    if message_id:
        kwargs["id"] = message_id
    name = payload.get("name")
    if name:
        kwargs["name"] = name
    return AIMessage(**kwargs)


def _llm_invoke_worker(
    output_queue,
    *,
    model_name: str,
    base_url: str,
    temperature: float,
    serialized_messages: list[dict[str, Any]],
    config: dict[str, Any] | None,
):
    try:
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )
        messages = [_deserialize_message(item) for item in serialized_messages]
        response = llm.invoke(messages, stream=False, config=config)
        output_queue.put(("ok", _serialize_response(response)))
    except Exception as exc:
        output_queue.put(("error", repr(exc)))


def invoke_llm_with_hard_timeout(
    *,
    model_name: str,
    messages,
    label: str,
    config: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
    temperature: float = 0,
    base_url: str | None = None,
):
    if timeout_seconds is None:
        timeout_seconds = int(getattr(Config, "TIMEOUT", 10) or 10)
    timeout_seconds = max(int(timeout_seconds), 1)
    serialized_messages = [_serialize_message(message) for message in messages]
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    process = ctx.Process(
        target=_llm_invoke_worker,
        kwargs={
            "output_queue": output_queue,
            "model_name": model_name,
            "base_url": base_url or Config.OLLAMA_BASE_URL,
            "temperature": temperature,
            "serialized_messages": serialized_messages,
            "config": config or {},
        },
        daemon=True,
    )
    process.start()

    try:
        status, payload = output_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        raise TimeoutError(f"{label} timed out after {timeout_seconds}s") from exc
    finally:
        output_queue.close()
        output_queue.join_thread()

    process.join(timeout=1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)

    if status == "error":
        raise RuntimeError(f"{label} failed: {payload}")

    return _build_ai_message(payload)
