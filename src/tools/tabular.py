import json
import os
from io import StringIO

import pandas as pd
from langchain_ollama import ChatOllama

from src.core.config import Config
from src.core.user_settings import get_setting
from src.tools.code_executor import execute_python_in_sandbox

SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
RESULT_MARKER = "__NEXUS_TABULAR_RESULT__"


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
        "Matplotlib is available as `plt`.\n"
        "Two helper functions are available:\n"
        "- emit_text(summary: str)\n"
        "- emit_plot(summary: str)\n"
        "Rules:\n"
        "- Return only raw Python code. No markdown fences.\n"
        "- Do not read files, write files, or use network access.\n"
        "- Use only the already-loaded DataFrame `df` plus pandas/numpy/matplotlib.\n"
        "- Do not call print(). Use emit_text() or emit_plot() exactly once as the final step.\n"
        "- If useful, compute summary statistics, groupbys, correlations, or derived metrics.\n"
        "- If the user asks for a chart, graph, histogram, trendline, scatter plot, bar chart, line chart, or pie chart, create the chart with matplotlib and finish with emit_plot().\n"
        "- If the user does not ask for a chart, do not create a plot; finish with emit_text().\n"
        "- Keep the summary concise and user-facing.\n\n"
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
        "import base64\n"
        "import json\n"
        "import os\n"
        "import pandas as pd\n"
        "os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from io import BytesIO, StringIO\n"
        f"RESULT_MARKER = {json.dumps(RESULT_MARKER)}\n"
        f"_DF_JSON = {json.dumps(df_json)}\n"
        "df = pd.read_json(StringIO(_DF_JSON), orient='records')\n"
        "def emit_text(summary):\n"
        "    payload = {'kind': 'text', 'summary': str(summary)}\n"
        "    print(f'{RESULT_MARKER}{json.dumps(payload)}')\n"
        "def emit_plot(summary):\n"
        "    buffer = BytesIO()\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(buffer, format='png', bbox_inches='tight')\n"
        "    buffer.seek(0)\n"
        "    image_b64 = base64.b64encode(buffer.read()).decode('ascii')\n"
        "    plt.close('all')\n"
        "    payload = {'kind': 'plot', 'summary': str(summary), 'image_base64': image_b64}\n"
        "    print(f'{RESULT_MARKER}{json.dumps(payload)}')\n"
    )
    sandbox_code = f"{bootstrap_code}\n{generated_code.strip()}\n"
    return execute_python_in_sandbox(sandbox_code)


def extract_tabular_result_payload(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            payload_str = line[len(RESULT_MARKER):]
            return json.loads(payload_str)
    return None


def format_tabular_result_content(
    user_query: str,
    abs_path: str,
    df: pd.DataFrame,
    generated_code: str,
    payload: dict | None,
    raw_stdout: str,
    stderr: str,
    timed_out: bool,
    exit_code: int,
) -> str:
    framing_header = (
        "The following content was generated by loading the user's local tabular file on the host, "
        "reconstructing it as a pandas DataFrame named `df` inside the sandbox, and running generated "
        "analysis code against it.\n"
        "Answer the user's original question using the sandbox results as your source.\n"
        "─────────────────────────────────────────\n"
    )

    parts = [
        framing_header,
        f"User query: {user_query}",
        f"File: {abs_path}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        f"Generated analysis code:\n{generated_code}",
        "",
    ]

    if payload:
        summary = payload.get("summary", "").strip()
        if summary:
            parts.append(f"Analysis summary:\n{summary}")
            parts.append("")

    raw_stdout = raw_stdout.strip()
    if raw_stdout:
        parts.append(f"Sandbox output:\n{raw_stdout}")
        parts.append("")
    if stderr.strip():
        parts.append(f"Sandbox errors:\n{stderr.strip()}")
        parts.append("")
    if not payload and not raw_stdout and not stderr.strip() and not timed_out and exit_code == 0:
        parts.append("Sandbox completed with no structured output.")
        parts.append("")
    if timed_out:
        parts.append("Sandbox execution timed out.")
    elif exit_code != 0 and not stderr.strip():
        parts.append(f"Sandbox exited with code {exit_code}.")

    return "\n".join(part for part in parts if part is not None).strip()
