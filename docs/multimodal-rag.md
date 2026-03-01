# Multimodal RAG Setup

Nexus now supports additive multimodal ingestion and retrieval using ONNX Runtime GPU when the CLIP-style ONNX model files are available.

## Environment

- Set `MULTIMODAL_EMBEDDINGS_ENABLED=true` to enable multimodal ingestion/search.
- Put the ONNX model files under `models/clip_onnx/` by default, or point `MULTIMODAL_EMBED_MODEL_DIR` to a different directory.
- The embedder expects:
  - `text_encoder.onnx`
  - `vision_encoder.onnx`
  - tokenizer files in the same folder or a `tokenizer/` subfolder

## GPU Runtime

- Install `onnxruntime-gpu`.
- Ensure the machine has a working CUDA runtime that matches the ONNX Runtime GPU build.
- By default Nexus requests `CUDAExecutionProvider`, then falls back to existing text-only search if the multimodal embedder cannot initialize.

## Helper Script

Use `python scripts/download_clip_onnx.py <base-url>` to download model artifacts from a stable host into the configured model directory.
