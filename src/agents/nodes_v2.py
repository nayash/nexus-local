import json
import time
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.agents.contracts import (
    EvidenceBundle,
    EvidenceItem,
    IntentPacket,
    ManagerDecision,
    WorkerResult,
    WorkerTask,
    extract_json_object,
)
from src.agents.state import AgentState
from src.agents.utils import trim_messages
from src.core.config import Config
from src.core.llm_timeout import invoke_llm_with_hard_timeout
from src.core.model_routing import (
    AUX_TASK_MANAGER_INTENT,
    AUX_TASK_MANAGER_REVIEW,
    get_model_for_task,
    log_model_selection,
)
from src.core.user_settings import get_setting
from src.tools.local import execute_local_retrieval_task_v2, get_nexus_identity_response
from src.tools.registry import analyze_tabular_file_tool
from src.tools.search import search_web
from src.tools.tool_results import extract_final_response


_MANAGER_INTENT_PROMPT = """You are the manager agent for a local-first assistant.

Classify the latest user request and return ONLY valid JSON.
Schema:
{
  "primary_intent": "local_content|local_metadata|local_catalog|web|identity|tabular|hybrid_local_web|general_chat",
  "normalized_query": "string",
  "requires_hybrid_local": true|false,
  "confidence": "high|medium|low",
  "rationale": "short reason"
}

Rules:
- Favor local intents when the user asks about their files, notes, ideas, or documents.
- Use local_catalog when the user wants a file or document inventory, such as listing files in a workspace/folder, counting files, or showing which documents are available.
- Use local_metadata when the user is trying to identify a specific file or answer metadata about one or a few files.
- Use local_content when the user wants facts, explanation, summary, or excerpts from inside document contents.
- Use identity for questions about Nexus itself.
- Use tabular only when focused file is tabular or query explicitly asks dataframe-style analytics.
- Use hybrid_local_web when both private and public knowledge are required.
- Use general_chat for non-retrieval conversation.
"""


_MANAGER_REVIEW_PROMPT = """You are the manager-review agent.

Given the evidence and task history, decide whether to dispatch another worker or synthesize now.
Return ONLY valid JSON with this schema:
{
    "action": "dispatch|synthesize",
  "continue_reasoning": true|false,
  "next_task": {
    "worker": "local_retrieval_worker|local_catalog_worker|web_retrieval_worker|identity_worker|tabular_worker|none",
    "objective": "string",
    "query": "string",
    "mode": "semantic_answer|document_lookup|catalog|full_document|hybrid|",
    "required_evidence": ["string"]
  },
  "reason": "short reason"
}

Rules:
- If evidence is sufficient, synthesize.
- If intent is local and evidence is weak, dispatch local_retrieval_worker.
- If intent is hybrid and only local evidence exists, dispatch web_retrieval_worker.
- Do not loop forever.
"""


_SYNTHESIS_PROMPT = """You are Nexus.

Use the provided evidence to answer the user query.
Requirements:
- Answer the user directly and accurately.
- Do not mention internal routing, agents, or hidden reasoning.
- If evidence is missing, say what is missing and provide the best grounded answer available.
- Do not output JSON.
"""


_llm_cache: dict[str, ChatOllama] = {}


def _phase_config(phase: str) -> dict:
    normalized = (phase or "").strip().lower()
    return {
        "metadata": {"nexus_phase": normalized},
        "tags": [f"nexus_phase:{normalized}"],
    }


def _get_llm(model_name: str | None = None) -> ChatOllama:
    model_name = (model_name or get_setting("model_name", "llama3.1")).strip()
    if model_name not in _llm_cache:
        _llm_cache[model_name] = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=Config.OLLAMA_BASE_URL,
        )
    return _llm_cache[model_name]


def _invoke_llm_with_timeout(llm: ChatOllama, messages, *, label: str, config: dict, timeout_seconds: int | None = None):
    if timeout_seconds is None:
        timeout_seconds = int(getattr(Config, "TIMEOUT", 10) or 10)
    return invoke_llm_with_hard_timeout(
        model_name=llm.model,
        messages=messages,
        label=label,
        config=config,
        timeout_seconds=timeout_seconds,
        temperature=getattr(llm, "temperature", 0),
        base_url=getattr(llm, "base_url", Config.OLLAMA_BASE_URL),
    )


def _last_user_query(messages: list) -> str:
    for message in reversed(messages or []):
        if getattr(message, "type", "") == "human":
            return getattr(message, "content", "") or ""
    return ""


def _invoke_contract(model_cls, system_prompt: str, user_payload: dict, phase: str):
    task_name = AUX_TASK_MANAGER_INTENT if getattr(model_cls, "__name__", "") == "IntentPacket" else AUX_TASK_MANAGER_REVIEW
    model_name = get_model_for_task(task_name)
    log_model_selection(task_name, model_name)
    llm = _get_llm(model_name)
    content = json.dumps(user_payload, ensure_ascii=True)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]
    started_at = time.perf_counter()
    print(
        "manager contract start | "
        f"phase={phase} | model={model_name} | "
        f"schema={getattr(model_cls, '__name__', str(model_cls))}"
    )
    print(f"manager contract payload full | {content}")
    contract_timeout = (
        int(getattr(Config, "MANAGER_INTENT_TIMEOUT", 30) or 30)
        if getattr(model_cls, "__name__", "") == "IntentPacket"
        else int(getattr(Config, "TIMEOUT", 10) or 10)
    )
    print(
        "manager contract timeout | "
        f"schema={getattr(model_cls, '__name__', str(model_cls))} | seconds={contract_timeout}"
    )
    response = _invoke_llm_with_timeout(
        llm,
        messages,
        label=f"manager contract {getattr(model_cls, '__name__', str(model_cls))}",
        config=_phase_config(phase),
        timeout_seconds=contract_timeout,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "manager contract end | "
        f"phase={phase} | schema={getattr(model_cls, '__name__', str(model_cls))} | "
        f"elapsed_ms={elapsed_ms}"
    )
    parsed = extract_json_object(getattr(response, "content", "") or "")
    try:
        return model_cls.model_validate(parsed)
    except ValidationError as exc:
        repair_started_at = time.perf_counter()
        print(
            "manager contract repair start | "
            f"phase={phase} | schema={getattr(model_cls, '__name__', str(model_cls))}"
        )
        repair_messages = messages + [
            AIMessage(content=getattr(response, "content", "") or ""),
            HumanMessage(
                content=(
                    "Your JSON was invalid for the required schema. "
                    f"Validation errors: {exc.errors()}. "
                    "Return corrected JSON only."
                )
            ),
        ]
        repaired = _invoke_llm_with_timeout(
            llm,
            repair_messages,
            label=f"manager contract repair {getattr(model_cls, '__name__', str(model_cls))}",
            config=_phase_config(phase),
            timeout_seconds=contract_timeout,
        )
        repair_elapsed_ms = round((time.perf_counter() - repair_started_at) * 1000, 1)
        print(
            "manager contract repair end | "
            f"phase={phase} | schema={getattr(model_cls, '__name__', str(model_cls))} | "
            f"elapsed_ms={repair_elapsed_ms}"
        )
        parsed_repaired = extract_json_object(getattr(repaired, "content", "") or "")
        try:
            return model_cls.model_validate(parsed_repaired)
        except ValidationError:
            return model_cls()


def _default_local_mode(intent_packet: IntentPacket) -> str:
    if intent_packet.primary_intent == "local_catalog":
        return "catalog"
    if intent_packet.primary_intent == "local_metadata":
        return "document_lookup"
    if intent_packet.primary_intent == "hybrid_local_web" or intent_packet.requires_hybrid_local:
        return "hybrid"
    return "semantic_answer"


def _dedupe_source_metadata(items: List[dict]) -> List[dict]:
    deduped = []
    seen = set()
    for item in items:
        key = (str(item.get("title", "")), str(item.get("url", "")), str(item.get("type", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _bundle_from_state(state: AgentState) -> EvidenceBundle:
    raw_bundle = state.get("evidence_bundle") or {}
    try:
        return EvidenceBundle.model_validate(raw_bundle)
    except ValidationError:
        return EvidenceBundle()


def _compact_text(value: str, limit: int = 280) -> str:
    text = " ".join((value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _compact_worker_history(history: list[dict], limit: int = 3) -> list[dict]:
    compact = []
    for item in (history or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "worker": item.get("worker", ""),
                "status": item.get("status", ""),
                "summary": _compact_text(str(item.get("summary", "") or ""), limit=240),
                "source_count": len(item.get("source_metadata") or []),
                "evidence_count": len(item.get("evidence") or []),
            }
        )
    return compact


def _compact_worker_summaries(summaries: list[str], limit: int = 3) -> list[str]:
    return [_compact_text(str(item or ""), limit=220) for item in (summaries or [])[-limit:]]


def _compact_evidence_preview(bundle: EvidenceBundle, limit: int = 4) -> list[dict]:
    preview = []
    for item in bundle.items[:limit]:
        preview.append(
            {
                "source_type": item.source_type,
                "title": _compact_text(item.title, limit=80),
                "score": item.score,
                "snippet": _compact_text(item.snippet, limit=180),
            }
        )
    return preview


def _append_worker_result(
    state: AgentState,
    worker_result: WorkerResult,
    manager_next_node: str = "manager_review",
) -> dict:
    bundle = _bundle_from_state(state)
    bundle.items.extend(worker_result.evidence)
    bundle.source_metadata.extend(worker_result.source_metadata)
    if worker_result.summary:
        bundle.worker_summaries.append(worker_result.summary)
    bundle.source_metadata = _dedupe_source_metadata(bundle.source_metadata)

    history_item = worker_result.model_dump(mode="json")
    return {
        "task_history": [history_item],
        "evidence_bundle": bundle.model_dump(mode="json"),
        "manager_next_node": manager_next_node,
    }


def _task_from_intent(intent: IntentPacket, query: str, focused_file: str) -> WorkerTask:
    if focused_file and focused_file.lower().endswith((".csv", ".tsv", ".xlsx", ".xls")):
        return WorkerTask(
            worker="tabular_worker",
            objective="Analyze the focused tabular file for the user query.",
            query=query,
            mode="semantic_answer",
        )

    mapping = {
        "local_content": "local_retrieval_worker",
        "local_metadata": "local_retrieval_worker",
        "local_catalog": "local_catalog_worker",
        "web": "web_retrieval_worker",
        "identity": "identity_worker",
        "tabular": "tabular_worker",
        "hybrid_local_web": "local_retrieval_worker",
        "general_chat": "none",
    }
    worker = mapping.get(intent.primary_intent, "none")
    return WorkerTask(
        worker=worker,
        objective=f"Handle intent: {intent.primary_intent}",
        query=query,
        mode=_default_local_mode(intent),
    )


def manager_intent_node(state: AgentState):
    started_at = time.perf_counter()
    print("manager_intent node start")
    trimmed = trim_messages(state.get("messages") or [], max_messages=12)
    print(f"manager_intent trimmed history | messages={len(trimmed)}")
    query = _last_user_query(trimmed)
    focused_file = (state.get("focused_file") or "").strip()
    workspace_id = (state.get("workspace_id") or "").strip()
    query_preview = " ".join((query or "").split())[:200]
    print(
        "manager_intent inputs | "
        f"query={query_preview!r} | focused_file={focused_file or 'none'} | "
        f"workspace_id={workspace_id or 'global'}"
    )

    payload = {
        "query": query,
        "focused_file": focused_file,
        "workspace_id": workspace_id,
        "conversation_tail": [
            {
                "type": getattr(message, "type", ""),
                "content": (getattr(message, "content", "") or "")[:300],
            }
            for message in trimmed[-5:]
        ],
    }
    print(
        "manager_intent payload summary | "
        f"tail_messages={len(payload['conversation_tail'])} | "
        f"tail_types={[item.get('type') for item in payload['conversation_tail']]}"
    )
    intent_packet = _invoke_contract(IntentPacket, _MANAGER_INTENT_PROMPT, payload, phase="decision")
    print(
        "manager_intent classified | "
        f"primary_intent={intent_packet.primary_intent} | "
        f"normalized_query={intent_packet.normalized_query!r} | "
        f"requires_hybrid_local={intent_packet.requires_hybrid_local} | "
        f"confidence={intent_packet.confidence}"
    )

    task = _task_from_intent(intent_packet, query, focused_file)
    next_node = task.worker if task.worker != "none" else "response_synthesizer"
    print(
        "manager_intent task | "
        f"worker={task.worker} | mode={task.mode or 'none'} | next_node={next_node}"
    )

    result = {
        "intent_packet": intent_packet.model_dump(mode="json"),
        "current_task": task.model_dump(mode="json"),
        "manager_hop_count": 0,
        "manager_next_node": next_node,
        "evidence_bundle": EvidenceBundle().model_dump(mode="json"),
    }
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "manager_intent node end | "
        f"intent={intent_packet.primary_intent} | next_node={next_node} | elapsed_ms={elapsed_ms}"
    )
    return result


def local_retrieval_worker_node(state: AgentState):
    started_at = time.perf_counter()
    print("local_retrieval_worker node start")
    task = WorkerTask.model_validate(state.get("current_task") or {})
    query = task.query or _last_user_query(state.get("messages") or [])
    file_filter = (state.get("focused_file") or "").strip()
    workspace_id = (state.get("workspace_id") or "").strip()
    mode = task.mode or "semantic_answer"

    result_payload = execute_local_retrieval_task_v2(
        query=query,
        file_filter=file_filter,
        workspace_id=workspace_id,
        mode=mode,
    )
    worker_result = WorkerResult.model_validate(result_payload)
    result = _append_worker_result(state, worker_result)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "local_retrieval_worker node end | "
        f"status={worker_result.status} | evidence={len(worker_result.evidence)} | elapsed_ms={elapsed_ms}"
    )
    return result


def local_catalog_worker_node(state: AgentState):
    started_at = time.perf_counter()
    print("local_catalog_worker node start")
    task = WorkerTask.model_validate(state.get("current_task") or {})
    query = task.query or _last_user_query(state.get("messages") or [])
    file_filter = (state.get("focused_file") or "").strip()
    workspace_id = (state.get("workspace_id") or "").strip()

    result_payload = execute_local_retrieval_task_v2(
        query=query,
        file_filter=file_filter,
        workspace_id=workspace_id,
        mode="catalog",
    )
    worker_result = WorkerResult.model_validate(result_payload)
    result = _append_worker_result(state, worker_result)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "local_catalog_worker node end | "
        f"status={worker_result.status} | evidence={len(worker_result.evidence)} | elapsed_ms={elapsed_ms}"
    )
    return result


def web_retrieval_worker_node(state: AgentState):
    started_at = time.perf_counter()
    print("web_retrieval_worker node start")
    task = WorkerTask.model_validate(state.get("current_task") or {})
    query = task.query or _last_user_query(state.get("messages") or [])
    results = search_web(query=query, category="general", time_range="")

    evidence = []
    metadata = []
    snippets = []
    for index, item in enumerate(results[:6], start=1):
        snippet = (item.content or "").strip()
        snippets.append(f"[{index}] {item.title}: {snippet}")
        evidence.append(
            EvidenceItem(
                source_type="web",
                title=item.title,
                url=item.url,
                snippet=snippet[:600],
                score=max(0.0, 1.0 - (0.1 * (index - 1))),
            )
        )
        metadata.append(
            {
                "title": item.title,
                "url": item.url,
                "type": "web",
            }
        )

    summary = "\n".join(snippets) if snippets else "No relevant web evidence found."
    worker_result = WorkerResult(
        worker="web_retrieval_worker",
        status="ok" if snippets else "empty",
        summary=summary[:5000],
        proposed_answer="",
        evidence=evidence,
        source_metadata=metadata,
    )
    result = _append_worker_result(state, worker_result)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "web_retrieval_worker node end | "
        f"status={worker_result.status} | evidence={len(worker_result.evidence)} | elapsed_ms={elapsed_ms}"
    )
    return result


def identity_worker_node(state: AgentState):
    started_at = time.perf_counter()
    print("identity_worker node start")
    task = WorkerTask.model_validate(state.get("current_task") or {})
    query = task.query or _last_user_query(state.get("messages") or [])
    content, metadata = get_nexus_identity_response(query)
    final_response = extract_final_response(metadata)
    summary = final_response or content

    evidence = [
        EvidenceItem(
            source_type="local",
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=(summary or "")[:1200],
            score=1.0,
        )
        for item in metadata
        if isinstance(item, dict) and item.get("type") != "final_response"
    ]

    worker_result = WorkerResult(
        worker="identity_worker",
        status="ok" if summary else "empty",
        summary=(summary or "No identity content available.")[:5000],
        proposed_answer=final_response or "",
        evidence=evidence,
        source_metadata=[
            item
            for item in metadata
            if isinstance(item, dict) and item.get("type") != "final_response"
        ],
    )
    result = _append_worker_result(state, worker_result)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "identity_worker node end | "
        f"status={worker_result.status} | evidence={len(worker_result.evidence)} | elapsed_ms={elapsed_ms}"
    )
    return result


def tabular_worker_node(state: AgentState):
    started_at = time.perf_counter()
    print("tabular_worker node start")
    task = WorkerTask.model_validate(state.get("current_task") or {})
    query = task.query or _last_user_query(state.get("messages") or [])
    focused_file = (state.get("focused_file") or "").strip()
    if not focused_file:
        worker_result = WorkerResult(
            worker="tabular_worker",
            status="error",
            summary="No focused tabular file is available for analysis.",
            proposed_answer="No focused tabular file is available for analysis.",
            evidence=[],
            source_metadata=[],
        )
        result = _append_worker_result(state, worker_result)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        print(
            "tabular_worker node end | "
            f"status={worker_result.status} | evidence={len(worker_result.evidence)} | elapsed_ms={elapsed_ms}"
        )
        return result

    try:
        content, metadata = analyze_tabular_file_tool.func(focused_file, query)
        metadata = [item for item in (metadata or []) if isinstance(item, dict)]
        plot_artifacts = [item for item in metadata if item.get("type") == "plot" and item.get("image_base64")]
        local_sources = [item for item in metadata if item.get("type") in {"local", "plot"}]
        if plot_artifacts:
            plot_summary = str(plot_artifacts[0].get("summary", "") or "").strip()
            if plot_summary:
                content = f"Generated requested plot.\n\n{plot_summary}"
            else:
                content = "Generated requested plot."
        status = "ok"
    except Exception as exc:
        content = f"Failed to analyze focused tabular file: {exc}"
        local_sources = [{"title": focused_file.split("/")[-1], "url": focused_file, "type": "local"}]
        status = "error"

    worker_result = WorkerResult(
        worker="tabular_worker",
        status=status,
        summary=content[:5000],
        proposed_answer="",
        evidence=[
            EvidenceItem(
                source_type="local",
                title=str((local_sources[0] if local_sources else {}).get("title", "Tabular Analysis")),
                url=str((local_sources[0] if local_sources else {}).get("url", focused_file)),
                snippet=content[:1200],
                score=1.0 if status == "ok" else 0.0,
            )
        ],
        source_metadata=local_sources,
    )
    result = _append_worker_result(state, worker_result)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "tabular_worker node end | "
        f"status={worker_result.status} | evidence={len(worker_result.evidence)} | elapsed_ms={elapsed_ms}"
    )
    return result


def manager_review_node(state: AgentState):
    started_at = time.perf_counter()
    print("manager_review node start")
    hop_count = int(state.get("manager_hop_count") or 0) + 1
    if hop_count >= 3:
        result = {
            "manager_hop_count": hop_count,
            "manager_next_node": "response_synthesizer",
        }
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        print(f"manager_review node end | action=hop_limit | elapsed_ms={elapsed_ms}")
        return result

    intent_packet = IntentPacket.model_validate(state.get("intent_packet") or {})
    current_task = WorkerTask.model_validate(state.get("current_task") or {})
    bundle = _bundle_from_state(state)
    history = state.get("task_history") or []
    compact_history = _compact_worker_history(history, limit=3)
    compact_summaries = _compact_worker_summaries(bundle.worker_summaries, limit=3)
    compact_evidence = _compact_evidence_preview(bundle, limit=4)

    payload = {
        "intent_packet": intent_packet.model_dump(mode="json"),
        "current_task": current_task.model_dump(mode="json"),
        "manager_hop_count": hop_count,
        "task_history": compact_history,
        "evidence_count": len(bundle.items),
        "worker_summaries": compact_summaries,
        "evidence_preview": compact_evidence,
    }
    print(
        "manager_review payload summary | "
        f"task_history={len(compact_history)} | worker_summaries={len(compact_summaries)} | "
        f"evidence_preview={len(compact_evidence)} | evidence_count={len(bundle.items)}"
    )
    decision = _invoke_contract(ManagerDecision, _MANAGER_REVIEW_PROMPT, payload, phase="decision")
    next_node = "response_synthesizer"
    next_task = decision.next_task

    if decision.action == "dispatch" and next_task.worker != "none":
        next_node = next_task.worker
    else:
        next_task = WorkerTask(worker="none")

    result = {
        "manager_hop_count": hop_count,
        "manager_next_node": next_node,
        "current_task": next_task.model_dump(mode="json"),
    }
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "manager_review node end | "
        f"action={decision.action} | next_node={next_node} | elapsed_ms={elapsed_ms}"
    )
    return result


def response_synthesizer_node(state: AgentState):
    started_at = time.perf_counter()
    print("response_synthesizer node start")
    messages = trim_messages(state.get("messages") or [], max_messages=14)
    query = _last_user_query(messages)
    intent_packet = IntentPacket.model_validate(state.get("intent_packet") or {})
    bundle = _bundle_from_state(state)

    evidence_lines = []
    for index, item in enumerate(bundle.items[:16], start=1):
        evidence_lines.append(
            f"[{index}] ({item.source_type}) {item.title} | {item.url}\n{item.snippet}"
        )

    synthesis_messages = [
        SystemMessage(content=_SYNTHESIS_PROMPT),
        HumanMessage(
            content=json.dumps(
                {
                    "query": query,
                    "intent": intent_packet.model_dump(mode="json"),
                    "worker_summaries": bundle.worker_summaries[-6:],
                    "evidence": evidence_lines,
                },
                ensure_ascii=True,
            )
        ),
    ]
    response = _invoke_llm_with_timeout(
        _get_llm(),
        synthesis_messages,
        label="response synthesizer",
        config=_phase_config("final"),
    )
    source_metadata = _dedupe_source_metadata(bundle.source_metadata)
    result = {
        "messages": [response],
        "sources": source_metadata,
        "final_draft": (getattr(response, "content", "") or ""),
    }
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "response_synthesizer node end | "
        f"sources={len(source_metadata)} | elapsed_ms={elapsed_ms}"
    )
    return result


def route_manager_next(state: AgentState):
    return (state.get("manager_next_node") or "response_synthesizer").strip() or "response_synthesizer"
