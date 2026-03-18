import os

from src.core.user_settings import get_setting


AUX_TASK_CHAT_TITLE = "chat_title"
AUX_TASK_MANAGER_INTENT = "manager_intent"
AUX_TASK_MANAGER_REVIEW = "manager_review"
AUX_TASK_LOCAL_RETRIEVAL_PLANNER = "local_retrieval_planner"

DEFAULT_AUX_TASKS = (
    AUX_TASK_CHAT_TITLE,
    AUX_TASK_MANAGER_INTENT,
    AUX_TASK_MANAGER_REVIEW,
    AUX_TASK_LOCAL_RETRIEVAL_PLANNER,
)

_AUX_TASK_DESCRIPTIONS = {
    AUX_TASK_CHAT_TITLE: "Chat title generation",
    AUX_TASK_MANAGER_INTENT: "Intent classification",
    AUX_TASK_MANAGER_REVIEW: "Manager review / next-step routing",
    AUX_TASK_LOCAL_RETRIEVAL_PLANNER: "Local retrieval mode planning",
}


def get_main_model_name(default: str = "llama3.1") -> str:
    return str(get_setting("model_name", default) or default).strip()


def get_aux_model_name() -> str:
    return str(get_setting("aux_model_name", "") or "").strip()


def get_enabled_aux_tasks() -> set[str]:
    raw = os.getenv("NEXUS_AUX_MODEL_TASKS", "").strip()
    if not raw:
        return set(DEFAULT_AUX_TASKS)
    return {item.strip() for item in raw.split(",") if item.strip()}


def should_use_aux_model(task_name: str) -> bool:
    aux_model = get_aux_model_name()
    if not aux_model:
        return False
    return task_name in get_enabled_aux_tasks()


def get_model_for_task(task_name: str, default: str = "llama3.1") -> str:
    if should_use_aux_model(task_name):
        return get_aux_model_name()
    return get_main_model_name(default=default)


def log_model_selection(task_name: str, model_name: str):
    main_model = get_main_model_name()
    aux_model = get_aux_model_name()
    route = "aux" if aux_model and model_name == aux_model else "main"
    print(
        "model routing | "
        f"task={task_name} | route={route} | model={model_name} | "
        f"main_model={main_model} | aux_model={aux_model or 'disabled'}"
    )


def describe_enabled_aux_tasks() -> str:
    enabled = get_enabled_aux_tasks()
    labels = [
        _AUX_TASK_DESCRIPTIONS[task]
        for task in DEFAULT_AUX_TASKS
        if task in enabled and task in _AUX_TASK_DESCRIPTIONS
    ]
    return ", ".join(labels)
