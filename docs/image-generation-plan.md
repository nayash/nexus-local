# Linux/Windows Image Generation Plan Using Local Diffusers

## Summary

The Linux support plan is to make image generation work locally now by using a dedicated `diffusers` backend for Linux and Windows, while keeping the Ollama image backend as a future optional path for macOS or later expansion.

That means:
- Nexus still exposes a single "generate image" capability to the user.
- The agent still auto-routes natural-language image requests.
- On Linux and Windows, the actual image generation is done by a local Python `diffusers` pipeline, not Ollama.
- Ollama remains a chat model provider and can remain a future image backend option, but it is not the primary path for this feature.

## How Local Diffusers Will Be Used

Local `diffusers` will sit behind a new Nexus image backend and will be invoked only when the agent decides the user is asking for image generation.

End-to-end flow:
1. The user types a natural-language request such as "generate an image of a mountain cabin in snow".
2. The agent detects image-generation intent and calls `generate_image_tool`.
3. `generate_image_tool` routes to the configured image backend, which on Linux/Windows will be `DiffusersImageBackend`.
4. The backend loads a local text-to-image pipeline using the configured model.
5. The pipeline runs on the local GPU with memory-saving settings appropriate for 8 GB VRAM.
6. The generated image is saved as a local PNG file under a managed output directory.
7. The tool returns:
   - a short textual summary for the assistant response
   - a visual artifact containing base64 image data for immediate inline rendering
   - metadata pointing to the saved file for persistence and later chat reload

Implementation behavior:
- The image pipeline should be initialized lazily on first use, not at app startup.
- The pipeline should be cached in memory after first load so repeated generations do not reload weights every time.
- The feature should be GPU-first.
- CPU fallback should be disabled by default, because it will be too slow for acceptable UX on most systems.
- The backend should retry once with reduced settings if GPU out-of-memory occurs.

## What the User Needs to Install

Image generation should be an optional dependency set, separate from the current base app install.

### Required System Prerequisites

1. A supported GPU and drivers
- For NVIDIA, the user needs a working NVIDIA GPU with a recent compatible driver.
- A full CUDA toolkit install is not necessarily required if PyTorch is installed from prebuilt CUDA wheels, but the driver must be compatible.

2. Enough VRAM
- The target baseline is an 8 GB consumer GPU.
- The first model and defaults should be selected specifically to fit that constraint.

3. Enough disk space
- The user needs free disk space for:
  - downloaded model weights
  - cached inference artifacts
  - generated image files
- Expect several GB of additional disk usage.

### Required Python Packages

The image feature should add an optional dependency set including:

- `torch` (CUDA-enabled build on Linux/Windows NVIDIA systems)
- `diffusers["torch"]`
- `transformers`
- `accelerate`
- `safetensors`

These should not be forced into the base installation for users who only want chat/search features.

### Model Weights

- The chosen text-to-image model must be downloaded locally the first time it is used, unless pre-cached.
- Nexus should check whether the configured model is already available and surface a clear setup/download state if not.

### Repo Constraint to Address

The current project file [pyproject.toml](/home/asutosh/Documents/projects/ml_projects/nexus-local/pyproject.toml) pins `requires-python = "=3.12"`.

That may be a Windows compatibility issue for local image generation, because PyTorch Windows support is typically more restrictive than Linux. The implementation plan should therefore treat Python version compatibility as part of the feature scope, not as a later cleanup item.

## Implementation Plan

### 1. Introduce a backend abstraction with Diffusers as the default local image engine

Create a new module such as `/home/asutosh/Documents/projects/ml_projects/nexus-local/src/tools/image_generation.py`.

Define these internal types:
- `ImageGenerationRequest`
- `ImageGenerationResult`
- `ImageGenerationBackend`

Backend implementations:
- `DiffusersImageBackend` for Linux and Windows
- `OllamaImageBackend` as optional/future
- `DisabledImageBackend` for clean failure when the feature is off

Default selection logic:
- Linux: `diffusers`
- Windows: `diffusers`
- macOS: `diffusers` by default, optional `ollama_image` if explicitly selected
- unsupported/misconfigured backend: structured error, not a crash

This keeps the rest of Nexus independent of the specific image engine.

### 2. Use a local 8 GB-friendly Diffusers model

For the first implementation, target an SDXL-turbo-class or SD 1.5-class model that is practical on 8 GB VRAM with memory optimizations enabled.

Recommended default for Linux/Windows:
- Start with an 8 GB-friendly `diffusers` pipeline using a small/fast text-to-image model.
- Keep the model identifier configurable instead of hard-coding one model deep in the tool logic.

Add config in [/home/asutosh/Documents/projects/ml_projects/nexus-local/src/core/config.py](/home/asutosh/Documents/projects/ml_projects/nexus-local/src/core/config.py):
- `IMAGE_BACKEND = "diffusers"`
- `DIFFUSERS_MODEL_ID = "<chosen default model id>"`
- `IMAGE_OUTPUT_DIR = <project>/data/generated_images`
- `IMAGE_MAX_IMAGES = 1`
- `IMAGE_TIMEOUT = 120`
- `IMAGE_DEVICE = "auto"`
- `IMAGE_DTYPE = "float16"`

Add user settings in [/home/asutosh/Documents/projects/ml_projects/nexus-local/src/core/user_settings.py](/home/asutosh/Documents/projects/ml_projects/nexus-local/src/core/user_settings.py):
- `image_backend`
- `diffusers_model_id`
- `enable_image_generation`

Implementation defaults for 8 GB stability:
- `torch.float16` on CUDA
- enable attention slicing
- enable VAE slicing/tiling where supported
- low inference steps for v1
- single image only
- conservative output resolution by default

### 3. Add the actual generator tool

In [/home/asutosh/Documents/projects/ml_projects/nexus-local/src/tools/registry.py](/home/asutosh/Documents/projects/ml_projects/nexus-local/src/tools/registry.py), add:
- `generate_image_tool`

Schema:
- `prompt: str`
- `size: Literal["1024x1024", "768x768", "1024x768", "768x1024"] = "1024x1024"`
- `negative_prompt: Optional[str] = None`
- `seed: Optional[int] = None`
- `steps: Optional[int] = None`

Behavior:
- validates the prompt
- chooses the configured backend
- calls the backend
- returns `(content, artifact)` using `response_format="content_and_artifact"`

Artifact shape:
- `type: "generated_image"`
- `mime: "image/png"`
- `image_base64`
- `file_path`
- `prompt`
- `model`
- `backend`

This makes image generation fit the same tool-execution pattern as existing tabular plots.

### 4. Auto-route image requests robustly

Keep the user experience as natural language only.

In [/home/asutosh/Documents/projects/ml_projects/nexus-local/src/agents/nodes.py](/home/asutosh/Documents/projects/ml_projects/nexus-local/src/agents/nodes.py):
- extend the system prompt so the model must call `generate_image_tool` when the user asks to create or generate an image
- add deterministic rescue logic for obvious image requests if the LLM answers with text instead of a tool call

Intent detection phrases:
- "generate an image"
- "create an image"
- "draw"
- "illustrate"
- "make a poster"
- "make a logo"
- "show me a picture of"

If a strong image-gen intent is detected and no tool call is emitted:
- rewrite to `generate_image_tool`
- preserve the full user prompt as the tool prompt

This mirrors the existing rescue logic already used for chatty tool calls.

### 5. Implement the Diffusers backend with proper local runtime checks

`DiffusersImageBackend` should:
- check that `torch`, `diffusers`, and required model files are available
- check whether CUDA is available
- if CUDA is unavailable, either:
  - run a slower CPU path only if explicitly allowed, or
  - return a clear "GPU required for current config" message

Recommended runtime behavior:
- prefer CUDA automatically
- if CUDA exists but VRAM is tight, reduce defaults:
  - lower resolution
  - lower step count
- if out-of-memory occurs:
  - retry once with smaller dimensions or fewer steps
  - then return a structured error

This is the Linux/Windows robustness layer that replaces the previous "unsupported Ollama" dead end.

### 6. Render generated images in the chat window using the existing base64 pattern

Yes, the current base64 transport path should be reused for immediate rendering.

Current reusable path:
- artifact capture in [/home/asutosh/Documents/projects/ml_projects/nexus-local/src/ui/agent_interface.py](/home/asutosh/Documents/projects/ml_projects/nexus-local/src/ui/agent_interface.py)
- image decoding/rendering in [/home/asutosh/Documents/projects/ml_projects/nexus-local/src/ui/components/chat_view.py](/home/asutosh/Documents/projects/ml_projects/nexus-local/src/ui/components/chat_view.py)

Required change:
- generalize from plot-only rendering to image artifact rendering

Implementation:
- keep `<nexus-plot>` support untouched
- add `<nexus-image mime="image/png">...</nexus-image>`
- parse both tags in chat rendering
- render both through `ft.Image`

Answer to your earlier question:
- yes, the existing base64 way can be used for display
- but it should be transport-only for the current reply, not the long-term storage format

### 7. Persist images as files plus metadata, not inline base64 in chat history

The current `messages` table in [/home/asutosh/Documents/projects/ml_projects/nexus-local/src/core/database.py](/home/asutosh/Documents/projects/ml_projects/nexus-local/src/core/database.py) only stores text, which is not suitable for large image payloads.

Add a new table:
- `message_artifacts`
- fields: `id`, `chat_id`, `message_id`, `type`, `mime`, `file_path`, `prompt`, `model`, `backend`, `created_at`

Storage rules:
- save generated PNGs under `data/generated_images/<chat_id>/<uuid>.png`
- save only artifact metadata in the DB
- on chat reload, reconstruct the visual elements from `message_artifacts`

This avoids database bloat and keeps chat history responsive.

### 8. Add setup and dependency handling for Linux/Windows

Because `diffusers` is a Python runtime dependency, plan for environment setup explicitly.

Add a startup validation path:
- check if `torch` and `diffusers` import successfully
- check if the configured model is locally cached
- if missing, show a clear setup/download message instead of failing at first generation request

The app should expose a user-facing status such as:
- "Image generation ready"
- "Model download required"
- "CUDA not detected"
- "Running in reduced mode"

Keep this separate from the existing Ollama readiness checks in startup.

Dependency strategy:
- keep current base app install unchanged
- add an optional image-generation dependency group for PyTorch + Diffusers stack
- do not force all users to install heavy ML packages if they do not use image generation

### 9. Keep Ollama as optional, not primary

Ollama remains useful in the architecture, but only as:
- the chat LLM
- a future or optional image backend on platforms where it is supported

Do not couple the Linux/Windows image-generation feature to Ollama support.

This directly addresses the gap you called out: Linux support should be real local generation, not "error because Ollama can't do it here."

## Public API / Interface Changes

New tool:
- `generate_image_tool(prompt, size="1024x1024", negative_prompt=None, seed=None, steps=None)`

New artifact type:
- `generated_image`

New UI tag:
- `<nexus-image mime="image/png">BASE64...</nexus-image>`

New config keys:
- `IMAGE_BACKEND`
- `DIFFUSERS_MODEL_ID`
- `IMAGE_OUTPUT_DIR`
- `IMAGE_MAX_IMAGES`
- `IMAGE_TIMEOUT`
- `IMAGE_DEVICE`
- `IMAGE_DTYPE`

New settings keys:
- `image_backend`
- `diffusers_model_id`
- `enable_image_generation`

New DB table:
- `message_artifacts`

## Test Cases and Acceptance Criteria

### Functional

1. On Linux, "Generate an image of a mountain cabin in snow" triggers `generate_image_tool`.
2. The tool uses the `diffusers` backend instead of any Ollama image API.
3. A generated image appears inline in the chat window.
4. Reloading the chat shows the same image from persisted artifact metadata.
5. Normal text questions still use the chat LLM and do not trigger image generation.

### Robustness

1. If CUDA is present, the backend runs on GPU using memory-saving options.
2. If GPU OOM occurs, the backend retries once with reduced settings.
3. If required Python packages are missing, the user gets a clear setup error.
4. If the selected model is not yet available locally, the app surfaces that requirement cleanly.
5. If image decoding fails in the UI, the chat does not crash.

### Regression

1. Existing plot rendering still works.
2. Existing source rendering still works and generated images are not listed as sources.
3. Existing tool routing and tabular analysis continue to work unchanged.

## Assumptions and Defaults

- Linux and Windows use local `diffusers` as the primary image backend
- The chat LLM remains Ollama-based
- v1 supports one generated image per request
- v1 uses GPU-first execution; CPU fallback is optional and not enabled by default
- v1 reuses base64 for immediate UI rendering, but persists image files on disk
- Ollama image generation is not the Linux/Windows implementation path
