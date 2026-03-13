#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${NEXUS_VENV_DIR:-$ROOT_DIR/.venv}"
INSTALL_STAMP="$VENV_DIR/.nexus-install-stamp"
RUNTIME_STAMP="$VENV_DIR/.nexus-runtime-stamp"
BOOTSTRAP_VERSION="2"
CONFIG_FILE="/etc/ld.so.conf.d/nexus-local-cudnn.conf"
TARGET_SONAME="libcudnn.so.9"

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup.sh [app|setup|run|doctor|gpu-runtime|check-gpu] [options]

Commands:
  app              Create/update the local virtualenv if needed, bootstrap the
                   runtime on first run, then launch the app. This is the
                   default command if none is provided.
  setup            Create/update the local virtualenv and run nexus-local setup.
  run              Create/update the local virtualenv if needed, then launch
                   the app without forcing setup again if it already completed.
  doctor           Run nexus-local doctor --check-multimodal inside the managed
                   virtualenv.
  gpu-runtime      Configure the system linker so ONNX Runtime can load cuDNN.
  check-gpu        Verify NVIDIA driver, cuDNN linker visibility, ONNX Runtime
                   providers, and the project's active multimodal embedder mode.

Options:
  --cudnn-dir PATH Use an explicit directory that contains libcudnn.so.9.
  --print-export   Print a temporary LD_LIBRARY_PATH export command instead of
                   writing system linker config.
  --force-install  Reinstall the Python environment and rerun nexus-local setup.
  --skip-setup     Skip nexus-local setup before launching the app.
  -h, --help       Show this help message.
EOF
}

log() {
  printf '[setup] %s\n' "$1"
}

warn() {
  printf '[setup] WARNING: %s\n' "$1" >&2
}

die() {
  printf '[setup] ERROR: %s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

find_cudnn_dir() {
  local explicit_dir="${1:-}"
  if [[ -n "$explicit_dir" ]]; then
    if [[ -f "$explicit_dir/$TARGET_SONAME" ]]; then
      printf '%s\n' "$explicit_dir"
      return 0
    fi
    die "The provided directory does not contain $TARGET_SONAME: $explicit_dir"
  fi

  local candidates=(
    "/usr/local/lib/ollama/mlx_cuda_v13"
    "/usr/local/cuda/lib64"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
    "/usr/lib"
    "/usr/local/lib"
  )

  local dir
  for dir in "${candidates[@]}"; do
    if [[ -f "$dir/$TARGET_SONAME" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
  done

  if command_exists find; then
    while IFS= read -r match; do
      if [[ -n "$match" ]]; then
        dirname "$match"
        return 0
      fi
    done < <(find /usr /usr/local -name "$TARGET_SONAME" 2>/dev/null)
  fi

  return 1
}

library_visible() {
  if ! command_exists ldconfig; then
    return 1
  fi
  ldconfig -p 2>/dev/null | grep -q "$TARGET_SONAME"
}

write_linker_config() {
  local lib_dir="$1"

  if [[ "$(id -u)" -eq 0 ]]; then
    printf '%s\n' "$lib_dir" > "$CONFIG_FILE"
    ldconfig
    return 0
  fi

  if command_exists sudo; then
    printf '%s\n' "$lib_dir" | sudo tee "$CONFIG_FILE" >/dev/null
    sudo ldconfig
    return 0
  fi

  return 1
}

resolve_python() {
  local candidate
  for candidate in python3.12 python3 python; do
    if command_exists "$candidate" && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

compute_project_stamp() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
from pathlib import Path
import hashlib

root = Path.cwd()
payload = []
for name in ("pyproject.toml", "scripts/setup.sh"):
    payload.append(Path(name).read_bytes())
digest = hashlib.sha256(b"\n".join(payload)).hexdigest()
print(digest)
PY
}

activate_managed_venv() {
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}

ensure_repo_root() {
  cd "$ROOT_DIR"
}

ensure_python_env() {
  local force_install="${1:-0}"
  local python_bin
  python_bin="$(resolve_python || true)"
  [[ -n "$python_bin" ]] || die "Python 3.12 is required. Install python3.12 and rerun this script."

  ensure_repo_root

  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating managed virtualenv at $VENV_DIR"
    "$python_bin" -m venv "$VENV_DIR"
  fi

  activate_managed_venv

  local current_stamp desired_stamp installed_stamp=""
  desired_stamp="${BOOTSTRAP_VERSION}:$(compute_project_stamp "$python_bin")"
  if [[ -f "$INSTALL_STAMP" ]]; then
    installed_stamp="$(<"$INSTALL_STAMP")"
  fi

  if [[ "$force_install" == "1" || ! -x "$VENV_DIR/bin/nexus-local" || "$installed_stamp" != "$desired_stamp" ]]; then
    log "Installing Nexus Local into the managed virtualenv"
    python -m pip install --upgrade pip
    pip install -e ".[full]"
    printf '%s\n' "$desired_stamp" > "$INSTALL_STAMP"
    rm -f "$RUNTIME_STAMP"
  else
    log "Python environment already matches the current project state"
  fi
}

ensure_runtime_setup() {
  local force_install="${1:-0}"
  local install_stamp=""
  local runtime_stamp=""

  [[ -f "$INSTALL_STAMP" ]] && install_stamp="$(<"$INSTALL_STAMP")"
  [[ -f "$RUNTIME_STAMP" ]] && runtime_stamp="$(<"$RUNTIME_STAMP")"

  if [[ "$force_install" == "1" || "$install_stamp" != "$runtime_stamp" ]]; then
    log "Running nexus-local setup"
    nexus-local setup
    printf '%s\n' "$install_stamp" > "$RUNTIME_STAMP"
  else
    log "Runtime bootstrap already completed; launching directly"
  fi
}

run_app_flow() {
  local force_install="${1:-0}"
  local skip_setup="${2:-0}"

  ensure_python_env "$force_install"
  if [[ "$skip_setup" != "1" ]]; then
    ensure_runtime_setup "$force_install"
  fi

  log "Starting Nexus Local"
  exec nexus-local run
}

run_setup_flow() {
  local force_install="${1:-0}"
  ensure_python_env "$force_install"
  ensure_runtime_setup "$force_install"
}

run_doctor_flow() {
  local force_install="${1:-0}"
  ensure_python_env "$force_install"
  log "Running nexus-local doctor"
  nexus-local doctor --check-multimodal
}

setup_gpu_runtime() {
  local explicit_cudnn_dir="$1"
  local print_export="$2"

  if library_visible; then
    log "$TARGET_SONAME is already available in the linker cache."
    if command_exists nvidia-smi; then
      log "NVIDIA driver detected."
    fi
    return 0
  fi

  local lib_dir
  lib_dir="$(find_cudnn_dir "$explicit_cudnn_dir" || true)"
  [[ -n "$lib_dir" ]] || die "Could not locate $TARGET_SONAME. Install cuDNN 9 first, then rerun this script."

  log "Found $TARGET_SONAME in: $lib_dir"

  if [[ "$print_export" -eq 1 ]]; then
    printf 'export LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$lib_dir"
    return 0
  fi

  if write_linker_config "$lib_dir"; then
    log "Configured linker path in $CONFIG_FILE"
    if library_visible; then
      log "$TARGET_SONAME is now available system-wide."
      return 0
    fi
    warn "The config file was written, but $TARGET_SONAME is still not visible in ldconfig output."
    return 1
  fi

  warn "Could not write system linker config automatically."
  warn "Run the following command in your shell as a temporary fallback:"
  printf 'export LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$lib_dir"
  return 1
}

check_gpu_runtime() {
  local python_bin
  python_bin="$(resolve_python || true)"

  log "Starting GPU runtime verification"

  if command_exists nvidia-smi; then
    log "nvidia-smi detected"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  else
    warn "nvidia-smi not found. NVIDIA driver may be unavailable."
  fi

  if library_visible; then
    log "$TARGET_SONAME is visible in the linker cache"
    ldconfig -p 2>/dev/null | grep "$TARGET_SONAME" || true
  else
    warn "$TARGET_SONAME is not visible in the linker cache"
    local lib_dir
    lib_dir="$(find_cudnn_dir "" || true)"
    if [[ -n "$lib_dir" ]]; then
      warn "Found $TARGET_SONAME on disk at: $lib_dir"
      warn "Run: bash scripts/setup.sh gpu-runtime --cudnn-dir $lib_dir"
    fi
  fi

  if [[ -z "$python_bin" ]]; then
    warn "Python interpreter not found; skipping ONNX Runtime checks."
    return 1
  fi

  log "Using Python interpreter: $python_bin"
  "$python_bin" -c '
import os
import sys

print("[setup] Python executable:", sys.executable)

try:
    import onnxruntime as ort
except Exception as exc:
    print(f"[setup] WARNING: Failed to import onnxruntime: {exc}")
    raise SystemExit(1)

providers = ort.get_available_providers()
print("[setup] onnxruntime available providers:", providers)

if "CUDAExecutionProvider" not in providers:
    print("[setup] WARNING: CUDAExecutionProvider is not exposed by this onnxruntime build")

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.embeddings.multimodal_onnx import get_multimodal_embedder
except Exception as exc:
    print(f"[setup] WARNING: Failed to import project multimodal embedder: {exc}")
    raise SystemExit(1)

embedder = get_multimodal_embedder(force_refresh=True)
if embedder is None:
    print("[setup] WARNING: Project multimodal embedder is unavailable")
    raise SystemExit(1)

active = getattr(embedder, "active_providers", [])
device = getattr(embedder, "device", "unknown")
print("[setup] project multimodal embedder device:", device)
print("[setup] project multimodal embedder active providers:", active)

if "CUDAExecutionProvider" in active:
    print("[setup] GPU multimodal embedding is active")
else:
    print("[setup] WARNING: Multimodal embedding is running without CUDA (likely CPU fallback)")
' || return 1

  return 0
}

COMMAND="app"
PRINT_EXPORT=0
EXPLICIT_CUDNN_DIR=""
FORCE_INSTALL=0
SKIP_SETUP=0

if [[ $# -gt 0 ]]; then
  case "$1" in
    app|setup|run|doctor|gpu-runtime|check-gpu)
      COMMAND="$1"
      shift
      ;;
    --cudnn-dir|--print-export|--force-install|--skip-setup|-h|--help)
      ;;
    *)
      die "Unknown command: $1"
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cudnn-dir)
      [[ $# -ge 2 ]] || die "Missing value for --cudnn-dir"
      EXPLICIT_CUDNN_DIR="$2"
      shift 2
      ;;
    --print-export)
      PRINT_EXPORT=1
      shift
      ;;
    --force-install)
      FORCE_INSTALL=1
      shift
      ;;
    --skip-setup)
      SKIP_SETUP=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

case "$COMMAND" in
  app|run)
    run_app_flow "$FORCE_INSTALL" "$SKIP_SETUP"
    ;;
  setup)
    run_setup_flow "$FORCE_INSTALL"
    ;;
  doctor)
    run_doctor_flow "$FORCE_INSTALL"
    ;;
  gpu-runtime)
    setup_gpu_runtime "$EXPLICIT_CUDNN_DIR" "$PRINT_EXPORT"
    ;;
  check-gpu)
    check_gpu_runtime
    ;;
  *)
    die "Unsupported command: $COMMAND"
    ;;
esac
