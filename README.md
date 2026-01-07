# Ullim VITS

Few-shot Korean voice cloning with VITS architecture

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

Ullim VITS is a few-shot voice cloning system based on the VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture, optimized for Korean language and A100 40GB GPU. The system enables high-quality voice synthesis with minimal reference audio samples.

### Key Features

- **Few-shot Voice Cloning**: Generate speech in new voices with just 3-10 audio samples
- **Korean Language Support**: Native Korean phoneme processing with g2pk2
- **Multi-speaker Training**: Trained on Zeroth-Korean dataset with 115 speakers
- **A100 Optimized**: Configuration optimized for A100 40GB GPU with mixed precision training
- **Modern Architecture**: VITS with HiFi-GAN decoder, normalizing flows, and stochastic duration prediction

## Features

- End-to-end text-to-speech synthesis
- Few-shot voice adaptation
- Multi-speaker voice cloning
- Korean text processing with automatic phoneme conversion
- WandB integration for experiment tracking
- Hydra-based configuration management
- Mixed precision training (FP16)

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Poetry

### Setup

1. Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/ullim-vits.git
cd ullim-vits
```

3. Install dependencies:

```bash
poetry install
```

4. For development (includes jupyter, pytest, etc.):

```bash
poetry install   --with dev
```

5. Activate environment:

```bash
poetry shell
```

## Quick Start

### 0. Environment Setup

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your WandB API key:

```
WANDB_API_KEY=your-wandb-api-key-here
```

Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

### 1. Data Preparation

**Option 1: Automatic Download (Recommended)**

Download and preprocess Zeroth-Korean dataset automatically:

```bash
poetry run ullim-preprocess
```

This will:

- Download Zeroth-Korean from HuggingFace
- Resample audio to 22050Hz
- Generate metadata files
- Compute dataset statistics

**Option 2: Manual Download**

If automatic download fails, manually download from [HuggingFace](https://huggingface.co/datasets/kresnik/zeroth_korean) and place in `data/zeroth/` directory, then run:

```bash
poetry run ullim-preprocess
```

### 2. Training

Start training with default configuration:

```bash
poetry run ullim-train

# Want hide nnpack errors
poetry run ullim-train 2>&1 | grep -v "NNPACK"
```

Or with custom config:

```bash
poetry run ullim-train model=vits_large train.batch_size=16
```

### 3. Inference

Generate speech from text:

```bash
poetry run ullim-infer \
  --checkpoint experiments/vits_fewshot_zeroth/checkpoints/checkpoint_100000.pt \
  --text "안녕하세요, 음성 합성 테스트입니다." \
  --output output.wav \
  --speaker_id 0
```

### 4. Few-shot Adaptation

Adapt model to a new voice with reference audio:

```python
from ullim_vits.inference.fewshot_adapter import FewShotAdapter

adapter = FewShotAdapter(config, checkpoint_path)
adapted_model = adapter.adapt(
    reference_audio_dir="path/to/reference/audio",
    reference_metadata="path/to/metadata.txt",
    target_speaker_id=999,
    n_epochs=100
)
adapter.save_adapted_model("adapted_model.pt")
```

## Usage

### Training

The training pipeline uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/vits_base.yaml`: Model architecture
- `configs/data/zeroth.yaml`: Dataset configuration
- `configs/train/default.yaml`: Training hyperparameters

#### Monitor Training

Training logs are automatically sent to WandB:

```bash
# View at https://wandb.ai/your-username/ullim-vits
```

#### Resume Training

```bash
poetry run ullim-train resume=experiments/vits_fewshot_zeroth/checkpoints/latest.pt
```

### Inference

Basic synthesis:

```python
from ullim_vits.inference.synthesizer import Synthesizer

synthesizer = Synthesizer(config, checkpoint_path)
audio = synthesizer.synthesize(
    text="음성 합성 테스트",
    speaker_id=0,
    noise_scale=0.667,
    length_scale=1.0
)
synthesizer.save_audio(audio, "output.wav")
```

### Configuration

Key hyperparameters in `configs/train/default.yaml`:

```yaml
batch_size: 32 # A100 40GB optimized
mixed_precision: true # FP16 training
gradient_clip_val: 1000.0
epochs: 1000

optimizer:
    generator:
        lr: 2.0e-4
    discriminator:
        lr: 2.0e-4

losses:
    mel_loss_weight: 45.0
    kl_loss_weight: 1.0
    feature_loss_weight: 2.0
```

## Project Structure

```
ullim-vits/
├── ullim_vits/              # Main package
│   ├── models/              # Neural network models
│   │   ├── vits/           # VITS components
│   │   ├── discriminator/  # MPD & MSD
│   │   ├── speaker_encoder/
│   │   └── common/         # Shared modules
│   ├── data/               # Dataset and data processing
│   ├── losses/             # Loss functions
│   ├── utils/              # Audio, text, alignment utilities
│   ├── training/           # Training loop and optimization
│   ├── inference/          # TTS and few-shot adaptation
│   └── cli/                # Command-line interfaces
├── configs/                # Hydra configurations
├── tools/                  # Preprocessing scripts
├── recipes/                # Dataset-specific recipes
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── experiments/            # Training outputs
```

See detailed documentation in `docs/` directory.

## Model Architecture

Ullim VITS implements the VITS architecture with the following components:

1. **Text Encoder (Prior Network)**
    - Transformer encoder with relative positional encoding
    - Normalizing flows for flexible prior distribution
    - Speaker conditioning via FiLM layers

2. **Posterior Encoder**
    - WaveNet-style residual blocks
    - Extracts latent representation from mel-spectrogram
    - Variational inference with speaker embedding

3. **Duration Predictor**
    - Stochastic duration prediction with flows
    - Monotonic Alignment Search (MAS) for alignment learning

4. **Decoder**
    - HiFi-GAN style vocoder
    - Multi-receptive field fusion
    - Transposed convolutions with residual blocks

5. **Discriminators**
    - Multi-Period Discriminator (MPD)
    - Multi-Scale Discriminator (MSD)

6. **Speaker Encoder**
    - CNN + GRU architecture
    - Extracts speaker embeddings from reference audio
    - Enables few-shot voice cloning

## Dataset

### Zeroth-Korean

- **Source**: [HuggingFace - kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean)
- **Speakers**: 115
- **Utterances**: 51,000+
- **Sampling Rate**: 22050 Hz (resampled)
- **Language**: Korean

### Custom Dataset

To use your own dataset, format your data as:

```
data/
├── train/
│   ├── audio/
│   │   ├── speaker1_001.wav
│   │   └── ...
│   └── metadata.txt
└── test/
    ├── audio/
    └── metadata.txt
```

Metadata format: `audio_filename|speaker_id|transcript`

```
speaker1_001.wav|speaker1|안녕하세요
```

## Training Details

### Hardware Requirements

- **Recommended**: NVIDIA A100 40GB
- **Minimum**: RTX 3090 24GB (reduce batch size to 16)
- **RAM**: 64GB+
- **Storage**: 100GB+ for dataset and checkpoints

### Training Time

On A100 40GB with default config:

- ~2-3 days for 1000 epochs on full Zeroth-Korean
- ~500k steps to convergence

### Hyperparameters

Model is configured for A100 40GB:

- Batch size: 32
- Mixed precision: FP16
- Gradient accumulation: 1
- Learning rate: 2e-4 (both G and D)
- Scheduler: Exponential (gamma=0.999875)

### Tips

- Start with smaller model (`vits_base`) for faster prototyping
- Use gradient checkpointing if running out of memory
- Monitor KL loss - should stabilize after 50k steps
- Duration predictor loss should decrease steadily
- Validation mel loss < 3.0 indicates good quality

## Results

Training results and audio samples will be added after initial training run.

### Metrics

- Mel-spectrogram reconstruction loss
- KL divergence
- Duration prediction accuracy
- MOS (Mean Opinion Score) - to be evaluated

## Citation & References

### VITS

```bibtex
@inproceedings{kim2021conditional,
  title={Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={International Conference on Machine Learning},
  pages={5530--5540},
  year={2021},
  organization={PMLR}
}
```

### Zeroth-Korean

```bibtex
@misc{zeroth_korean,
  title={Zeroth-Korean},
  author={Goodatlas},
  year={2017},
  url={https://github.com/goodatlas/zeroth}
}
```

### HiFi-GAN

```bibtex
@inproceedings{kong2020hifi,
  title={HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis},
  author={Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17022--17033},
  year={2020}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgements

- VITS implementation inspired by [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- HiFi-GAN decoder from [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- Zeroth-Korean dataset from [goodatlas/zeroth](https://github.com/goodatlas/zeroth)
- Korean phonemizer: g2pk2

## Documentation

Detailed documentation available in `docs/`:

- [Installation Guide](docs/installation.md)
- [Training Guide](docs/training.md)

## TODO

- [ ] Add pretrained model weights
- [ ] Add audio sample demos
- [ ] Implement emotion control
- [ ] Add voice conversion mode
- [ ] Multi-GPU training support
- [ ] ONNX export for deployment
- [ ] Real-time inference optimization
