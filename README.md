# LLM-CLIP: Cross-modal Image-Text Retrieval with Frozen LLM Text Encoder

Lightweight adaptation layers over a frozen Qwen2.5-1.5B text encoder for industrial technical manual retrieval. **R@1 = 38.15%** (+18% absolute over Chinese-CLIP LoRA baseline).

## Key Idea

Standard CLIP text encoders are capped at 77 tokens — insufficient for long-form technical documents. We replace the CLIP text tower with a frozen Qwen2.5-1.5B (512 token context, rich domain knowledge) and train only lightweight pooling and projection layers.

**Core designs**:
- **Multi-Head Attention Pooling**: A learnable `[AGG]` query vector attends over LLM token outputs via cross-attention, replacing simple mean pooling (+2.84% R@1).
- **Local-Global Patch Fusion**: Enhances ViT `[CLS]` representation with patch-level local features and residual projection for training stability (+3.32% R@1).

## Results

| Model | R@1 | R@5 | R@10 |
|-------|-----|-----|------|
| Chinese-CLIP LoRA | 20.14 | 40.76 | 53.32 |
| AltCLIP LoRA | 22.51 | 45.26 | 57.58 |
| **Qwen-Naive** (mean pooling) | 31.99 | 60.43 | 72.99 |
| **Qwen + MHA Pooling** | 34.83 | 59.00 | 69.91 |
| **Qwen + MHA + Patch (Ours)** | **38.15** | **63.27** | **71.33** |

> From the best external baseline (AltCLIP, 22.51%) to our best model (38.15%), an absolute gain of **15.64 percentage points**.

Length ablation reveals: LLM domain knowledge contributes ~13pp of the gain; extended context contributes ~1.7pp. The former dominates.

## Setup

```bash
pip install torch==2.4.0 transformers==4.44.0 peft==0.12.0 tqdm pillow torchvision
```

GPU memory ≥ 24 GB (Qwen2.5-1.5B + Chinese-CLIP ViT-B/16).

## Reproducibility

| Item | Setting |
|------|---------|
| PyTorch | 2.4.0 |
| CUDA | 12.1 |
| Transformers | 4.44.0 |
| PEFT | 0.12.0 |
| Random seed | Not set (default) |
| Optimizer | AdamW (weight decay = 1e-4) |
| Learning rate | 1e-4 (constant, no scheduler) |
| Warm-up | None |
| Batch size | 16 |
| Gradient accumulation | 2 steps (effective batch = 32) |
| Epochs | 20 |
| Mixed precision | AMP (autocast + GradScaler) |
| Temperature τ | Initialized at 0.07, learnable |
| LoRA (external baselines) | r=4, α=8, targets q_proj + v_proj |

## Data

Dataset follows `TechManualDataset` (see `dataset.py`):

```json
[{"img": "images/img_001.jpg", "original": "Engine disassembly steps...", "paraphrases": ["variant1", "variant2", "variant3"]}]
```

Images resized to 224×224 with CLIP normalization.

## Quick Start

```bash
# Best model
python experiments/exp1_main_results/mha_patch.py

# Baselines
python experiments/exp1_main_results/mha_pooling.py
python experiments/exp1_main_results/qwen_naive.py
python experiments/exp1_main_results/chinese_clip_lora.py

# Length ablation
python experiments/exp1_main_results/qwen_len77.py
python experiments/exp1_main_results/qwen_len256.py
```

## Structure

```
├── model_v2.py
├── dataset.py
├── train_joint.py
├── evaluate.py
└── experiments/exp1_main_results/
    ├── mha_patch.py              # ★ Best: MHA Pooling + Patch Fusion
    ├── mha_pooling.py            # MHA pooling baseline
    ├── mha_patch_nores.py        # Ablation: no residual connection
    ├── qwen_naive.py             # Plug-and-play baseline
    ├── naive_patch.py            # Mean pooling + Patch fusion
    ├── chinese_clip_lora.py      # External baseline
    ├── chinese_clip_zero_shot.py
    ├── qwen_len{77,128,256,512}.py  # Length sensitivity
    └── eval_by_length.py
```
