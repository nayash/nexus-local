# Nexus-Local

**Version:** 0.1.0 (Native Desktop Edition)
**Status:** DRAFT
**Project Codename:** Nexus-Local
**Target Platform:** Consumer Laptop (Nvidia 8GB VRAM)

---
<img width="1627" height="489" alt="Screenshot from 2026-03-01 00-27-59" src="https://github.com/user-attachments/assets/ee9d017c-0254-41f1-9ab5-031d5cc678ac" />

<img width="1915" height="1042" alt="Screenshot from 2026-02-27 19-46-35" src="https://github.com/user-attachments/assets/3712edfc-d692-4333-b1c8-f025b9af946e" />

<img width="1916" height="1040" alt="Screenshot from 2026-03-01 00-21-45" src="https://github.com/user-attachments/assets/1737f44b-7154-471e-aacd-c338b9fab3ee" />
---

## Quick Usage Guide
```bash
git clone https://github.com/nayash/nexus-local.git
cd nexus-local
bash scripts/setup.sh
```


## 1. Executive Summary
**Nexus-Local** is a privacy-first, hybrid AI Desktop Application that unifies **Open Web Search** and **Local Knowledge Retrieval**. Unlike cloud-based tools, Nexus runs entirely on your machine. It uses a stateful graph architecture (LangGraph) to intelligently route user intent, retrieve information from the best source (Web vs. Local), and synthesize answers using a local LLM.

**Key Pivot:** We are moving away from a browser-based UI (Chainlit) to a **Native Desktop Application (Flet)** to enable system-level features like global hotkeys, file system management, and local hardware integration.

---

## 2. Core Value Proposition
- **"The Private Perplexity":** A single search bar for your entire digital life.
- **Data Sovereignty:** Your financial PDFs, medical records, and codebases never leave your SSD.
- **Graph-Based Reasoning:** Uses a cyclic state machine to self-correct queries and verify answers.
- **Native OS Integration:** Can read, organize, and manage local files directly, not just "chat" with them.

---

## 3. Technical Architecture (8GB VRAM Budget)

### 3.1. The "VRAM Budget" Strategy
To prevent OOM (Out of Memory) crashes on consumer GPUs:

| Component | Technology | VRAM Allocation |
| :--- | :--- | :--- |
| **LLM Engine** | Ollama (`Llama-3.1-8B-Instruct` Q4_K_M) | ~5.5 GB |
| **Embedding Model** | `nomic-embed-text` (Quantized) | ~0.5 GB |
| **Context Window** | KV Cache (approx 4k tokens) | ~1.5 GB |
| **OS Overhead** | Display / System | ~0.5 GB |
| **Total** | | **~8.0 GB** |

### 3.2. Tech Stack
- **Frontend (The Body):** **Flet** (Python-based Flutter wrapper).
    - *Role:* Native Desktop UI, Window Management, Markdown Rendering, Streaming Text.
    - *Reasoning:* Allows packaging as a standalone `.exe` or `.app` without requiring a browser or hidden web server.
- **Orchestration (The Brain):** **LangGraph**.
    - *Role:* Manages the agent state, routing, and loops (e.g., Router -> Retriever -> Grader -> Generator).
- **LLM Server:** **Ollama**.
    - *Role:* Runs the GGUF models locally.
- **Vector DB:** **LanceDB**.
    - *Role:* Serverless, disk-based vector store (Zero-RAM overhead).
- **Web Search:** **SearXNG** (Dockerized).
    - *Role:* Aggregates results from Google/Bing anonymously.

---

## 4. Feature Set

### 4.1. Phase 1: The Foundation (Completed/Porting)
These features were verified in the prototype and are being ported to the Flet Desktop App.

1.  **Hybrid Search Router:** Automatically detects if a query needs "Web Search" (SearXNG), "Local Search" (LanceDB), or both.
2.  **Local RAG (Retrieval Augmented Generation):** Ingest PDF, TXT, and MD files into a local vector store and chat with them.
3.  **Streaming UI:** Real-time token generation with "Thought Process" visibility (collapsible step-by-step reasoning).
4.  **Source Citations:** Every claim is footnoted with a clickable link to the Web URL or Local File path.
5.  **Multi-turn Memory:** Maintains context across the conversation session.
6.  **Strict Citation Mode (Trust Engine):**
    * *Feature:* A "Professional Mode" that forces the LLM to say "I don't know" if it cannot find a specific source, preventing hallucination.
    * *Tech:* Adjusted System Prompt and Verification Node in LangGraph.

### 4.1.1. UI Design
1. Landing page should be minimalistic, modern, and sleek.
2. Chat interface should be like Gemini with side bar for conversation-history and settings, textbox at bottom for input with model selection dropdown to the bottom-right and an attach file button on the bottom-left and a send button on the center-right.
3. On click of settings button settings page should open up.
    3.1 In settings page there should be options to choose a directory or single file (for phase 1, supported formats are .txt, .pdf, .md, .csv, .json, .html, .htm, .xml or any text type file.)  which could be ingested into the local vector database for RAG for queries.
    3.2 There should be a button to clear the conversation history and the local vector database.

### 4.1.2 Low Level Features
1. Each ingested directory should have its own vector database and should be stored in a separate LanceDB folder in /data/lancedb/
2. 
    

### 4.2. Phase 2: The "Standout" Features (Next Roadmap)
These features leverage the **Native Desktop** advantage of Nexus Local.

1.  **Project Workspaces (Scoped RAG):**
    * *Feature:* Users can define "Projects" (e.g., "Thesis", "Tax 2024").
    * *Tech:* Nexus tags ingested files with `project_id`. Users select a workspace to chat *only* with those files, reducing noise.
2.  **Smart Daily Briefing (Automated Agents):**
    * *Feature:* A "Morning Report" agent that proactively searches the web for user-defined topics and checks local notes for updates.
    * *Tech:* Background cron job triggering a LangGraph workflow.
4.  **Active File Manager (Action over Text):**
    * *Feature:* "Smart Organizer" mode where Nexus watches a "Downloads" folder and auto-renames/moves files based on content (e.g., "Move AWS Invoice to /Finance").
    * *Tech:* Python `watchdog` library + OS file operations.
5.  **Secure Data Vault (Privacy Sandbox):**
    * *Feature:* A designated "Vault" folder where files are guaranteed to NEVER be sent to the Web Search tool or Cloud APIs (if switched to hybrid). User can chat with the contents from this vault.
    * *Tech:* Strict routing logic in the Graph.
6.  **Codebase Companion (Dev Context):**
    * *Feature:* Point Nexus at a local Code Repository. It indexes `.py`, `.js`, etc., and answers architectural questions like "Where is the auth logic?".
    * *Tech:* Specialized code-splitter chunking strategy.
7. **Multi-Modal File Ingestion**
    * *Feature:* Non-text based files can also be ingested like images, audio, video, etc. and can be used for RAG.
8. **The "Visual Context" Key (Multimodal Screen Chat)** - ChatGPT has a desktop app, but sending screenshots of your bank account or sensitive dashboards to the cloud is risky. Nexus can use local Vision models (like `Llama-3.2-Vision` or `Qwen-VL`) to "see" your screen instantly and privately.
    * *Feature:* A global hotkey (e.g., `Alt + Space`). When pressed, Nexus takes a snapshot of the active window and waits for a prompt.
    * *User Scenarios:*
        - *Data Entry:* Open a PDF invoice side-by-side with Excel. *Highlight invoice -> Hotkey -> "Extract the table into CSV format."*
        - *Debugging:* *Highlight error message in terminal -> Hotkey -> "Explain this error."*
    * *The "Gap":* Zero-latency, zero-privacy-risk visual analysis. It feels like an OS feature, not a chatbot website.
9. **Routine job to vectorize chat histories for future RAG**

---

## 5. Directory Structure (Suggested)

```text
nexus-local/
├── src/
│   ├── agents/          # LangGraph Logic
│   │   ├── graph.py     # Main StateMachine definition
│   │   ├── nodes.py     # Router, Retriever, Generator functions
│   │   └── state.py     # Pydantic State Schema
│   ├── ui/              # Flet Frontend
│   │   ├── main.py      # Main Desktop Window entry point
│   │   ├── chat.py      # Chat bubble components
│   │   └── settings.py  # Model/Profile configuration
│   ├── core/            # any code files core to the system
│   │   ├── config.py    # Env vars and constants
│   │   └── llm.py       #
│   ├── rag/             # RAG related code files
│   ├── tools/           # tools for Agents to use
│   ├── notebooks/       # for any ipynb files
```


### Run instruction for Flet:
```bash
#!/usr/bin/env bash
set -euo pipefail

############################################
current project init commands
############################################
echo "changing directory"
cd Documents/projects/ml_projects/nexus-local

echo "activating venv"
source .flet-venv/bin/activate

echo "starting Flet application"
PYTHONPATH=. flet run --recursive src/ui/main.py

---

## 6. Setup

### 6.1 Base App Setup

1. Create and activate the project virtual environment.
2. Install Nexus:

```bash
pip install .[full]
```

3. Bootstrap local runtime requirements:

```bash
nexus-local setup
```

4. Start the app:

```bash
nexus-local run
```

### 6.1.1 Diagnostics

Run non-mutating diagnostics:

```bash
nexus-local doctor --check-multimodal
```

Optional data directory override:

```bash
export NEXUS_DATA_DIR=/path/to/custom/nexus-data
```

### 6.2 Multimodal Local RAG Setup (ONNX + LanceDB)

Nexus supports additive multimodal ingestion and retrieval for:
- PDF (text + extracted images)
- DOCX
- TXT / MD
- CSV
- PNG / JPEG
- HTML / HTM

This feature uses:
- `onnxruntime-gpu` for GPU inference
- a CLIP-style ONNX model for text and image embeddings
- LanceDB for mixed text/image vector storage

#### Install the required packages

If you already installed from `requirements.txt`, the packages are included there. To install explicitly:

```bash
pip install onnxruntime-gpu transformers tokenizers pillow pymupdf python-docx beautifulsoup4
```

#### Verify ONNX Runtime GPU is available

Run:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output should include:

```text
CUDAExecutionProvider
```

If it does not, multimodal embeddings will disable themselves and Nexus will fall back to the existing text-only embedding path.

#### Configure the local GPU runtime (Linux)

If `CUDAExecutionProvider` is visible but session creation still fails with errors like `libcudnn.so.9: cannot open shared object file`, configure the linker path once from the repo root:

```bash
bash scripts/setup.sh gpu-runtime
```

This will locate `libcudnn.so.9`, add its directory to the system linker config, and run `ldconfig` so you do not need to export `LD_LIBRARY_PATH` manually on every shell.

If you know the exact cuDNN library directory, you can provide it explicitly:

```bash
bash scripts/setup.sh gpu-runtime --cudnn-dir /usr/local/lib/ollama/mlx_cuda_v13
```

#### Verify the full GPU stack

After the runtime setup, verify that the app can actually bind the multimodal embedder to CUDA:

```bash
bash scripts/setup.sh check-gpu
```

This checks:
- `nvidia-smi`
- `libcudnn.so.9` visibility in `ldconfig`
- ONNX Runtime provider availability
- whether the project's multimodal embedder is using `CUDAExecutionProvider` or CPU fallback

#### Create the local model directory

From the repo root:

```bash
mkdir -p models/clip_onnx
```

#### Download the ONNX model files

Use the Hugging Face ONNX export for CLIP:
- Model repo: `https://huggingface.co/Xenova/clip-vit-base-patch32`
- ONNX files: `https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx`

Download these exact files:

```bash
curl -L https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx -o models/clip_onnx/text_model.onnx
curl -L https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx -o models/clip_onnx/vision_model.onnx
```

#### Download the tokenizer files

Use the tokenizer files from the original CLIP model repo:
- Model repo: `https://huggingface.co/openai/clip-vit-base-patch32`

Download these exact files:

```bash
curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json -o models/clip_onnx/tokenizer.json
curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json -o models/clip_onnx/tokenizer_config.json
curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/special_tokens_map.json -o models/clip_onnx/special_tokens_map.json
curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json -o models/clip_onnx/vocab.json
curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt -o models/clip_onnx/merges.txt
```

After this, your folder should look like:

```text
models/clip_onnx/
  text_model.onnx
  vision_model.onnx
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.json
  merges.txt
```

#### Environment configuration

These settings are already loaded automatically from `.env` by `src/core/config.py`. You do not need to export them manually every time.

Current multimodal defaults:

```env
MULTIMODAL_EMBEDDINGS_ENABLED=true
MULTIMODAL_EMBED_MODEL_DIR=/home/asutosh/Documents/projects/ml_projects/nexus-local/models/clip_onnx
EMBEDDING_DEVICE=cuda
ORT_PROVIDER=CUDAExecutionProvider
```

#### Verify the embedder loads

Run:

```bash
python -c "from src.embeddings.multimodal_onnx import get_multimodal_embedder; e = get_multimodal_embedder(force_refresh=True); print(type(e).__name__ if e else 'FAILED')"
```

Expected result:

```text
MultimodalOnnxEmbedder
```

If it prints `FAILED`, check:
- `CUDAExecutionProvider` is available
- the ONNX files exist in `models/clip_onnx`
- the tokenizer files exist in `models/clip_onnx`
- the files were downloaded correctly

#### Optional helper script

You can also use:

```bash
python scripts/download_clip_onnx.py https://huggingface.co/openai/clip-vit-base-patch32/resolve/main
```

This downloads expected model files into the configured local model directory. It does not choose a source automatically; you must provide the base URL.

#### Multimodal ingestion test

To test with a sample folder containing mixed file types:

```bash
python scripts/test_multimodal_ingestion.py <sample-folder>
```

This will:
- ingest supported files into the multimodal LanceDB table
- run a few test queries
- print the top results with modality and citation metadata

---

## 7. Installation Paths

There are two supported ways to run Nexus Local from source.

### Path A — One-command repo setup and run

This is the primary user path after cloning the repository.

```bash
git clone https://github.com/nayash/nexus-local.git
cd nexus-local
bash scripts/setup.sh
```

What this does:
- creates a managed local virtualenv in `.venv` if needed
- installs or updates the project into that virtualenv
- runs `nexus-local setup` on the first run or after project changes
- starts the app

Second run behavior:
- if the environment is already prepared, `bash scripts/setup.sh` skips reinstall/bootstrap work and launches the app directly

Useful commands:

```bash
bash scripts/setup.sh doctor
bash scripts/setup.sh setup
bash scripts/setup.sh run
bash scripts/setup.sh --force-install
```

### Path B — Manual Python install

Use this if you want direct control over the Python environment.

```bash
git clone <your-repo-url>
cd nexus-local
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .[full]
nexus-local setup
nexus-local doctor --check-multimodal
nexus-local run
```

Notes:
- Path A is the recommended end-user source install flow.
- Path B is the advanced/manual fallback.
