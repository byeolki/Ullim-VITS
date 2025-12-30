# Installation Guide

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (A100 40GB recommended)
- **RAM**: 64GB+ recommended
- **Storage**: 100GB+ free space

### Software

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **Poetry**: Latest version

## Step-by-Step Installation

### 1. Install Poetry

#### Linux/macOS

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Verify Installation

```bash
poetry --version
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/ullim-vits.git
cd ullim-vits
```

### 3. Install Dependencies

#### Basic Installation

```bash
poetry install
```

#### With Development Tools

```bash
poetry install --with dev
```

This installs:

- pytest for testing
- black for code formatting
- ruff for linting
- jupyter for notebooks
- mypy for type checking

### 4. Activate Environment

```bash
poetry shell
```

### 5. Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import ullim_vits; print(ullim_vits.__version__)"
```

### 6. Configure WandB (Optional but Recommended)

WandB is used for experiment tracking and visualization.

#### Get API Key

1. Sign up at [wandb.ai](https://wandb.ai)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

#### Set Up Environment

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```bash
WANDB_API_KEY=wandb-api-key
```

#### Configure Entity

Edit `configs/config.yaml` and set your WandB username:

```yaml
wandb:
    enabled: true
    project: "ullim-vits"
    entity: "your-username"
```

#### Alternative: Login via CLI

```bash
poetry run wandb login
```

#### Disable WandB

```bash
poetry run ullim-train wandb.enabled=false
```

Or set environment variable:

```bash
export WANDB_MODE=disabled
```

## CUDA Setup

### Check CUDA Version

```bash
nvcc --version
nvidia-smi
```

### Install Specific PyTorch Version

If you need a specific CUDA version:

```bash
# For CUDA 11.8
poetry add torch==2.5.0+cu118 torchaudio==2.5.0+cu118 --source https://download.pytorch.org/whl/cu118

# For CUDA 12.1
poetry add torch==2.5.0+cu121 torchaudio==2.5.0+cu121 --source https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### Poetry Installation Issues

If Poetry command not found:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### CUDA Out of Memory

Reduce batch size in `configs/train/default.yaml`:

```yaml
batch_size: 16 # or 8
```

### g2pk2 Installation Fails

Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# macOS
brew install python
```

### ImportError for torchaudio

Reinstall with specific version:

```bash
poetry remove torchaudio
poetry add torchaudio==2.5.0
```

## Next Steps

After installation:

1. Download dataset: `poetry run ullim-preprocess`
2. Start training: `poetry run ullim-train`
3. See [Training Guide](training.md) for details
