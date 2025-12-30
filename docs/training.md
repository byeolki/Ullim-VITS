# Training Guide

## Quick Start

### 1. Prepare Data

Download and preprocess Zeroth-Korean:

```bash
poetry run ullim-preprocess
```

### 2. Configure Training

Edit `configs/train/default.yaml` if needed:

```yaml
batch_size: 32
epochs: 1000
mixed_precision: true
```

### 3. Start Training

```bash
poetry run ullim-train
```

## Configuration

### Hydra Configuration System

Ullim VITS uses Hydra for flexible configuration management.

#### Base Configuration

`configs/config.yaml`:

```yaml
defaults:
    - model: vits_base
    - data: zeroth
    - train: default

experiment_name: "vits_fewshot_zeroth"
output_dir: "experiments/${experiment_name}"
```

#### Override from Command Line

```bash
# Change model size
poetry run ullim-train model=vits_large

# Adjust batch size
poetry run ullim-train train.batch_size=16

# Multiple overrides
poetry run ullim-train model=vits_large train.batch_size=16 train.mixed_precision=false
```

## Model Configurations

### Base Model (Default)

`configs/model/vits_base.yaml`:

- Hidden channels: 192
- Transformer layers: 6
- Attention heads: 2
- ~50M parameters

### Large Model

`configs/model/vits_large.yaml`:

- Hidden channels: 256
- Transformer layers: 8
- Attention heads: 4
- ~100M parameters

## Training Strategies

### From Scratch

```bash
poetry run ullim-train
```

### Resume Training

```bash
poetry run ullim-train resume=experiments/vits_fewshot_zeroth/checkpoints/latest.pt
```

### Fine-tuning

1. Load pretrained checkpoint
2. Reduce learning rate
3. Train for fewer epochs

```bash
poetry run ullim-train \
  resume=pretrained_model.pt \
  train.optimizer.generator.lr=5e-5 \
  train.epochs=100
```

## Monitoring

### WandB Integration

Training automatically logs to WandB:

- Loss curves
- Audio samples
- Gradient norms
- Learning rates

Configure in `configs/train/default.yaml`:

```yaml
wandb:
    enabled: true
    project: "ullim-vits"
    entity: "your-username"
```

### Checkpoints

Saved every 5000 steps by default:

```
experiments/vits_fewshot_zeroth/checkpoints/
├── checkpoint_5000.pt
├── checkpoint_10000.pt
└── latest.pt
```

### Logs

Training logs saved to:

```
experiments/vits_fewshot_zeroth/
├── train.log
└── .hydra/
    └── config.yaml
```

## Optimization Tips

### Memory Optimization

**Reduce batch size:**

```bash
poetry run ullim-train train.batch_size=16
```

**Gradient accumulation:**

```yaml
gradient_accumulation_steps: 2
```

**Gradient checkpointing** (to be implemented)

### Speed Optimization

**Mixed precision:**

```yaml
mixed_precision: true # Default
```

**Increase workers:**

```yaml
num_workers: 16
```

**Persistent workers:**

```yaml
persistent_workers: true
```

## Multi-GPU Training

Coming soon - DDP support

## Hyperparameter Tuning

### Learning Rate

Start with default 2e-4, adjust if:

- Loss explodes → reduce to 1e-4
- Slow convergence → increase to 5e-4

### Loss Weights

Adjust in `configs/train/default.yaml`:

```yaml
losses:
    mel_loss_weight: 45.0 # Reconstruction quality
    kl_loss_weight: 1.0 # Posterior-prior matching
    feature_loss_weight: 2.0 # Discriminator features
    gen_loss_weight: 1.0 # Generator adversarial
    duration_loss_weight: 1.0 # Duration prediction
```

### Duration Predictor

If alignment issues:

- Increase `duration_loss_weight`
- Check MAS convergence in logs

## Evaluation

### Validation

Runs automatically every 1000 steps:

```yaml
eval_every: 1000
```

### Generate Samples

During training or after:

```bash
poetry run ullim-infer \
  --checkpoint experiments/vits_fewshot_zeroth/checkpoints/checkpoint_100000.pt \
  --text "테스트 문장입니다" \
  --output sample.wav
```

## Troubleshooting

### NaN Loss

- Reduce learning rate
- Enable gradient clipping (default: 1000.0)
- Check data preprocessing

### Poor Audio Quality

- Train longer (>100k steps)
- Increase mel_loss_weight
- Check discriminator training

### Slow Alignment

- Increase duration_loss_weight
- Verify MAS implementation
- Check phoneme quality

## Custom Dataset Training

1. Prepare data in correct format
2. Create config: `configs/data/custom.yaml`
3. Train: `poetry run ullim-train data=custom`

See README for data format details.
