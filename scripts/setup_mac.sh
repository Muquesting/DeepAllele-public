#!/usr/bin/env bash
set -euo pipefail

# Setup local macOS venv and install deps for DeepAllele
# Usage: bash scripts/setup_mac.sh

# Resolve repo root (parent of this script directory)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${REPO_ROOT}/.venv"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install requirements and package in editable mode
python -m pip install -r requirements-mac.txt
python -m pip install -e .

python - <<'PY'
import torch
print('Torch version:', torch.__version__)
print('MPS available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
PY

echo "[OK] macOS dev environment ready in .venv"
