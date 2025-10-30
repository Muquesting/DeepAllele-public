#!/usr/bin/env bash
set -euo pipefail

# Gadi GPU environment setup for DeepAllele
# Usage: bash scripts/setup_gadi.sh [cu118|cu121]
# - Expects $PROJECT to be defined (NCI project directory). Falls back to $HOME if unset.

CUDA_SPEC="${1:-cu118}"

# Load site modules if available (non-fatal if absent)
if command -v module &>/dev/null || [[ -n "${MODULESHOME:-}" ]]; then
  module load python/3.11 || module load python/3.10 || true
  module load cuda/11.8 || module load cuda/12.1 || true
fi

# Base dir to place venv and repos
BASE_DIR="${PROJECT:-$HOME}"
VENV_DIR="$BASE_DIR/venvs/deepallele"
REPO_DIR="$BASE_DIR/repos/DeepAllele-public"
mkdir -p "$(dirname "$VENV_DIR")" "$BASE_DIR/repos"

# Create/activate venv
python -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Get code
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone git@github.com:Muquesting/DeepAllele-public.git "$REPO_DIR"
fi

# Install deps
pip install -r "$REPO_DIR/requirements-linux-gpu.txt"
# Install Torch matching CUDA
case "$CUDA_SPEC" in
  cu118|cu121) INDEX_URL="https://download.pytorch.org/whl/${CUDA_SPEC}" ;;
  *) echo "Unknown CUDA spec '$CUDA_SPEC' (expected cu118|cu121)" >&2; exit 1 ;;
 esac
pip install torch torchvision torchaudio --index-url "$INDEX_URL"

# Install package
pip install -e "$REPO_DIR"

# Sanity check GPU
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY

echo "[OK] Gadi environment ready in $VENV_DIR (repo at $REPO_DIR)"
