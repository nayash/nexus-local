import json
import os
from io import StringIO

import pandas as pd
from langchain_ollama import ChatOllama

from src.core.config import Config
from src.core.user_settings import get_setting
from src.tools.code_executor import execute_python_in_sandbox

SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def load_tabular_dataframe(file_path: str) -> tuple[pd.DataFrame, str]:
    abs_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"The file '{abs_path}' was not found on the local filesystem.")

    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in SUPPORTED_TABULAR_EXTENSIONS:
        raise ValueError(f"The file '{abs_path}' is not a supported tabular format.")

    if ext == ".csv":
        df = pd.read_csv(abs_path)
    elif ext == ".tsv":
        df = pd.read_csv(abs_path, sep="\t")
    else:
        df = pd.read_excel(abs_path)

    return df, abs_path


def _build_dataframe_context(df: pd.DataFrame) -> str:
    dtypes = "\n".join(f"- {col}: {dtype}" for col, dtype in df.dtypes.items())
    sample_rows = df.head(5).to_json(orient="records", date_format="iso")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols_str = ", ".join(numeric_cols) if numeric_cols else "None"
    return (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Columns and dtypes:\n{dtypes}\n"
        f"Numeric columns: {numeric_cols_str}\n"
        f"Sample rows (JSON): {sample_rows}"
    )


def generate_tabular_analysis_code(user_query: str, df: pd.DataFrame) -> str:
    model_name = get_setting("model_name", "llama3.1")
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        base_url=Config.OLLAMA_BASE_URL,
        headers={"X-Thinking-Mode": "enable"},
    )

    prompt = (
        "You generate Python code for tabular analysis.\n"
        "A pandas DataFrame named `df` will already exist in the runtime.\n"
        "Rules:\n"
        "- Return only raw Python code. No markdown fences.\n"
        "- Do not read files, write files, or use network access.\n"
        "- Use only the already-loaded DataFrame `df` plus pandas/numpy.\n"
        "- Print concise results that directly answer the user's request.\n"
        "- If useful, compute summary statistics, groupbys, correlations, or derived metrics.\n"
        "- Avoid plots and large dumps of the full dataframe.\n"
        "- The final answer must be produced via print().\n\n"
        f"User request:\n{user_query}\n\n"
        f"DataFrame context:\n{_build_dataframe_context(df)}\n"
    )

    response = llm.invoke(prompt)
    raw_content = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw_content, list):
        raw_content = "".join(str(part) for part in raw_content)

    code = _strip_code_fences(str(raw_content))
    if not code:
        raise ValueError("The code-generation model returned empty analysis code.")

    return code


def execute_tabular_analysis(df: pd.DataFrame, generated_code: str):
    df_json = df.to_json(orient="records", date_format="iso")
    bootstrap_code = (
        "import pandas as pd\n"
        "from io import StringIO\n"
        f"_DF_JSON = {json.dumps(df_json)}\n"
        "df = pd.read_json(StringIO(_DF_JSON), orient='records')\n"
    )
    sandbox_code = f"{bootstrap_code}\n{generated_code.strip()}\n"
    return execute_python_in_sandbox(sandbox_code)
