#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# UllimVITS server environment setup script
# Target: Ubuntu 20.04+, CUDA 11.8 / 12.1 / 12.4
# ──────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Python version check ───────────────────
info "Checking Python version..."
PYTHON_BIN=$(command -v python3.11 || command -v python3.10 || command -v python3 || true)
[[ -z "$PYTHON_BIN" ]] && error "Python 3.10+ not found. Install it first."
PY_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 || ("$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10) ]]; then
    error "Python 3.10+ required, found $PY_VER"
fi
info "Python $PY_VER OK"

# ── 2. CUDA version detection ─────────────────
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    info "CUDA $CUDA_VERSION detected"
elif command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    info "CUDA $CUDA_VERSION detected (via nvidia-smi)"
else
    warn "No CUDA found — will install CPU-only PyTorch"
fi

# Map to PyTorch wheel suffix
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
if [[ "$CUDA_MAJOR" -ge 12 ]]; then
    TORCH_SOURCE="https://download.pytorch.org/whl/cu121"
    CUDA_TAG="cu121"
elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
    TORCH_SOURCE="https://download.pytorch.org/whl/cu118"
    CUDA_TAG="cu118"
else
    TORCH_SOURCE="https://download.pytorch.org/whl/cpu"
    CUDA_TAG="cpu"
fi
info "Using PyTorch wheel: $CUDA_TAG"

# ── 3. Poetry install / update ────────────────
if ! command -v poetry &>/dev/null; then
    info "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | "$PYTHON_BIN" -
    export PATH="$HOME/.local/bin:$PATH"
    # Persist PATH in common shell rc files
    for RC in ~/.bashrc ~/.zshrc; do
        if [[ -f "$RC" ]] && ! grep -q 'poetry' "$RC"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
        fi
    done
else
    info "Poetry $(poetry --version) already installed"
fi

# ── 4. Configure Poetry: in-project venv ──────
poetry config virtualenvs.in-project true

# ── 5. Install CPU deps first (no torch yet) ──
info "Installing base dependencies (excluding torch)..."
poetry install --no-root

# ── 6. Swap torch to CUDA build ───────────────
if [[ "$CUDA_TAG" != "cpu" ]]; then
    info "Replacing torch/torchaudio with $CUDA_TAG builds..."
    VENV_PIP="$(poetry env info --path)/bin/pip"
    "$VENV_PIP" install --upgrade \
        "torch==2.5.0+${CUDA_TAG}" \
        "torchaudio==2.5.0+${CUDA_TAG}" \
        --index-url "$TORCH_SOURCE"
fi

# ── 7. Install project package ────────────────
poetry install --only-root

# ── 8. .env setup ─────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.example .env
    warn ".env created from .env.example — set WANDB_API_KEY before training"
fi

# ── 9. Directories ────────────────────────────
mkdir -p experiments data/raw data/processed

# ── 10. Smoke test ────────────────────────────
info "Running smoke test..."
poetry run python - <<'EOF'
import torch, torchaudio, ullim_vits
print(f"torch      : {torch.__version__}")
print(f"torchaudio : {torchaudio.__version__}")
print(f"CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
print("ullim_vits imported OK")
EOF

info "Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit .env and set WANDB_API_KEY"
echo "    2. poetry run ullim-preprocess   # download & preprocess Zeroth dataset"
echo "    3. poetry run ullim-train        # start training"
echo ""
