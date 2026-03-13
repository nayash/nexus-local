#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
DEV_VENV="${DEV_VENV:-.venv-release}"
E2E_VENV="${E2E_VENV:-/tmp/nexus-e2e}"
SKIP_E2E="${SKIP_E2E:-0}"
KEEP_VENVS="${KEEP_VENVS:-0}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"

echo "[local-release] root: $ROOT_DIR"
echo "[local-release] python: $PYTHON_BIN"
echo "[local-release] keep venvs: $KEEP_VENVS"
echo "[local-release] keep build artifacts: $KEEP_ARTIFACTS"

safe_rm_dir() {
  local target="$1"
  if [[ -z "$target" || "$target" == "/" || "$target" == "." ]]; then
    return
  fi
  if [[ -d "$target" ]]; then
    rm -rf "$target"
  fi
}

cleanup() {
  if [[ "$KEEP_VENVS" == "1" ]]; then
    echo "[local-release] KEEP_VENVS=1, skipping venv cleanup"
    return
  fi

  # Deactivate currently active venv, if any, before deleting folders.
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate || true
  fi

  safe_rm_dir "$DEV_VENV"
  safe_rm_dir "$E2E_VENV"
  echo "[local-release] cleaned venvs: $DEV_VENV $E2E_VENV"

  if [[ "$KEEP_ARTIFACTS" == "1" ]]; then
    echo "[local-release] KEEP_ARTIFACTS=1, keeping build/dist"
    return
  fi

  safe_rm_dir "build"
  safe_rm_dir "dist"
  echo "[local-release] cleaned artifacts: build dist"
}

trap cleanup EXIT

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[local-release] ERROR: $PYTHON_BIN not found."
  echo "Install Python 3.12 or run with PYTHON_BIN=<path-to-python3.12>."
  exit 1
fi

if ! command -v flet >/dev/null 2>&1; then
  echo "[local-release] WARNING: 'flet' not found in current shell."
  echo "It will be available after dependency installation in $DEV_VENV."
fi

echo "[local-release] creating dev venv: $DEV_VENV"
"$PYTHON_BIN" -m venv "$DEV_VENV"
source "$DEV_VENV/bin/activate"

echo "[local-release] installing package + full extras"
pip install -U pip
pip install -e ".[full]"
pip install pytest

echo "[local-release] running packaging and test gates"
python -m pip wheel --no-deps . -w dist
PYTHONPATH=. pytest -q tests --maxfail=1 -k "not watcher_lifecycle"

echo "[local-release] building Linux desktop bundle"
mkdir -p release/linux
flet build linux src --yes --no-rich-output --output release/linux/linux-bundle
tar -C release/linux -czf release/linux/nexus-local-linux-bundle.tar.gz linux-bundle

echo "[local-release] collecting diagnostics and checksums"
(nexus-local doctor --check-multimodal || true) > release/linux/doctor-linux.txt
cp docs/release-bootstrap.md release/linux/release-bootstrap.md
cp dist/*.whl release/linux/

python - <<'PY'
import hashlib
from pathlib import Path

root = Path("release/linux")
lines = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(root)
        lines.append(f"{digest}  {rel}")
(root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
print("[local-release] wrote", root / "SHA256SUMS.txt")
PY

if [[ "$SKIP_E2E" == "1" ]]; then
  echo "[local-release] skipping end-user e2e install (SKIP_E2E=1)"
  echo "[local-release] artifacts ready at release/linux"
  exit 0
fi

echo "[local-release] running end-user wheel install smoke test in $E2E_VENV"
rm -rf "$E2E_VENV"
"$PYTHON_BIN" -m venv "$E2E_VENV"
source "$E2E_VENV/bin/activate"

pip install -U pip
pip install release/linux/*.whl
nexus-local --help >/dev/null
(nexus-local doctor || true) > /tmp/nexus-e2e-doctor.txt

echo "[local-release] e2e smoke test done."
echo "[local-release] artifacts ready at release/linux"
echo "[local-release] set KEEP_VENVS=1 and/or KEEP_ARTIFACTS=1 if you want to keep local test outputs."
