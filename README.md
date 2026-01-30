# Product Requirement Document (PRD): Nexus-Local

**Version:** 2.0 (Native Desktop Edition)
**Status:** DRAFT
**Project Codename:** Nexus-Local
**Target Platform:** Consumer Laptop (Nvidia 8GB VRAM)

---

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
|   |── rag/             # RAG related code files
|   |── tools/           # tools for Agents to use
|   |── notebooks/       # for any ipynb files
```


### Run instruction for Flet:
PYTHONPATH=. flet run --recursive src/ui/main.py