# Release Bootstrap Guide

## End-user commands

After installing Nexus Local, run:

```bash
nexus-local setup
nexus-local run
```

Diagnostics:

```bash
nexus-local doctor --check-multimodal
```

## Notes

- `setup` bootstraps Ollama, required models, Pyodide runtime, Docker sandbox image, and ONNX assets.
- `doctor` is non-mutating and reports readiness for all major features.
- Use `NEXUS_DATA_DIR` to override the default per-user application data directory.

