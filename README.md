# Fine-Tuning CLIP for Remote Sensing Image Classification

> Parameter-efficient fine-tuning of OpenCLIP ViT-B/32 on the RSICD dataset using LoRA adapters, with a semi-supervised extension combining supervised contrastive learning and SimCLR-style self-supervised learning on unlabeled aerial imagery.

---

## Results at a Glance

| Model | Zero-shot Accuracy | Macro F1 | I→T R@10 | T→I R@10 |
|---|---|---|---|---|
| Pre-trained CLIP (no fine-tuning) | 75.75% | 0.7562 | — | — |
| Supervised fine-tuning (LoRA) | 73.28% | 0.7320 | — | — |
| Semi-supervised (λ = 0.1) | 71.91% | 0.7073 | 0.3550 | 0.3808 |
| Semi-supervised (λ = 0.5) | 69.08% | 0.6865 | 0.3532 | 0.3832 |
| Semi-supervised (λ = 0.9) | 70.27% | 0.6985 | 0.3038 | 0.3833 |

**CodaBench Leaderboard Score: 66%** (21-class hidden test set, zero-shot)

---

## Overview

This project was completed as part of the TDDE70 Deep Learning course at Linköping University. The goal is to adapt a large pre-trained vision-language model (CLIP) to the remote sensing domain using the [RSICD dataset](https://github.com/201528014227051/RSICD_optimal), and evaluate it on zero-shot classification and cross-modal retrieval tasks.

The key challenge is a taxonomy mismatch: RSICD contains 31 land-cover classes with five human-written captions per image, while the external leaderboard evaluates on 21 land-use categories with no overlap in class indices. Rather than training a classifier, we use RSICD captions to teach the model what aerial imagery looks like, then classify test images using text prompts at inference time — making zero-shot transfer the core strategy.

Two versions are implemented:

- **Basic version**: Fully supervised fine-tuning on labeled image-caption pairs (Split B)
- **Advanced version**: Semi-supervised fine-tuning combining labeled (Split B) and unlabeled (Split A) data

---

## Method

### Parameter-Efficient Fine-Tuning with LoRA

All pre-trained weights are frozen. Low-Rank Adaptation (LoRA) matrices are injected into the attention output projections and MLP layers (`c_fc`, `c_proj`, `out_proj`) of both the visual and text encoders:

```
W = W₀ + BA,   B ∈ R^(d×r),  A ∈ R^(r×k),  r = 8
```

This reduces trainable parameters to ~1.47M (0.97% of total), significantly reducing overfitting risk on the small labeled set.

### Supervised Loss (Split B)

Standard symmetric CLIP contrastive loss over batches of image-caption pairs:

```
L_sup = -1/2N Σ [ log(exp(vᵢᵀuᵢ/τ) / Σⱼ exp(vᵢᵀuⱼ/τ))
                + log(exp(uᵢᵀvᵢ/τ) / Σⱼ exp(uᵢᵀvⱼ/τ)) ]
```

### Self-Supervised Loss (Split A)

For each unlabeled image, two augmented views are generated and encoded. Embeddings are passed through a non-linear projection head (512 → 512 → 128) and the NT-Xent loss is applied:

```
L_unsup = -1/2M Σᵢ log( exp(sim(zᵢ, z_{j(i)})/τₛ) / Σ_{k≠i} exp(sim(zᵢ, zₖ)/τₛ) )
```

Augmentations include random resized cropping, horizontal/vertical flipping, color jitter, and random grayscale. Vertical flipping is included deliberately — aerial images have no canonical orientation.

### Combined Objective

```
L_total = L_sup + λ · L_unsup
```

`λ` controls the trade-off between cross-modal alignment and self-supervised visual regularization. Best results were obtained at `λ = 0.1`.

### Training Procedure

1. **Warm-up (3 epochs)**: Supervised-only training on Split B to stabilize cross-modal alignment before introducing the self-supervised signal
2. **Joint training (20 epochs)**: Combined loss, loop driven by Split A (136 batches/epoch), Split B cycled to match

---

## Project Structure

```
├── Deep_Learning_Project_1_.ipynb   # Main notebook (training + evaluation)
├── predictions_final.txt            # Leaderboard submission file
├── confusion_matrix.png             # Confusion matrix for supervised model
├── README.md
└── report/
    └── report.pdf                   # Final NeurIPS-format report
```

---

## Dataset

Download the RSICD dataset from the [official repository](https://github.com/201528014227051/RSICD_optimal) (requires Git LFS). You need three files:

| File | Description |
|---|---|
| `RSICD_images/` | All dataset images |
| `txtclasses_rsicd/` | Per-class text files mapping class names to image filenames |
| `dataset_rsicd.json` | Image metadata including captions and split assignments |

Dataset splits used in this project:

| Split | JSON tag | Size | Usage |
|---|---|---|---|
| Split A | `train` | 8,734 images | Unlabeled — self-supervised only |
| Split B | `val` | 1,094 images | Labeled — supervised training |
| Split C | `test` | 1,093 images | Evaluation only (never used in training) |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/clip-rsicd-finetuning
cd clip-rsicd-finetuning

# Install dependencies
pip install torch torchvision open_clip_torch peft scikit-learn matplotlib tqdm
```

Update the dataset paths in the notebook:

```python
img_dir  = '/path/to/RSICD_images/'
json_dir = '/path/to/dataset_rsicd.json'
```

---

## Reproducing Results

Open `Deep_Learning_Project_1_.ipynb` and run cells in order. The notebook is divided into the following sections:

1. **Model loading** — OpenCLIP ViT-B/32 with LAION-2B weights
2. **Dataset preparation** — Split B (supervised) and Split A (unlabeled)
3. **LoRA configuration** — PEFT setup and trainable parameter count
4. **Supervised fine-tuning** — Basic version training loop
5. **Zero-shot evaluation** — Classification, confusion matrix, F1
6. **Semi-supervised fine-tuning** — Advanced version with warm-up
7. **Retrieval evaluation** — Image-to-text and text-to-image Recall@K
8. **Leaderboard inference** — Predictions on hidden 21-class test set

---

## Zero-Shot Classification Setup

Classification uses prompt ensembling over four templates per class:

```python
templates = [
    "A satellite image of {}.",
    "An aerial image of {}.",
    "A remote sensing image of {}.",
    "An overhead view of {}.",
]
```

Class embeddings are the normalized average of all template embeddings. The predicted class is the one with the highest cosine similarity to the image embedding.

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.2+ |
| open_clip_torch | latest |
| peft | latest |
| scikit-learn | latest |
| torchvision | latest |
| matplotlib | latest |
| tqdm | latest |

Training was performed on a single NVIDIA GPU with 19.5GB VRAM. Mixed precision training (AMP) is enabled by default to reduce memory consumption.

---

## References

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021
- Ilharco et al., [OpenCLIP](https://doi.org/10.5281/zenodo.5143773), Zenodo 2021
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022
- Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), ICML 2020
- Lu et al., Exploring Models and Data for Remote Sensing Image Caption Generation, IEEE TGRS 2018

---

## Course

TDDE70 Deep Learning — Linköping University, Spring 2026
