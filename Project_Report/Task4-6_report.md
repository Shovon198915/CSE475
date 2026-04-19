# SSL Method Comparison — PM25Vision AQI Classification

**Backbone:** EfficientNet-B0 | **Pretrain:** 30 epochs | **Fine-tune:** 20 epochs | **Batch:** 32 (DINO: 16) | **Seed:** 42

---

## Architecture & Training Summary

| Property | BYOL | Barlow Twins | DINO |
|---|---|---|---|
| **Notebook** | `ssl-1-byol-bootstrap.ipynb` | `ssl-2-barlow-twins.ipynb` | `ssl-3-dino.ipynb` |
| **Paradigm** | Bootstrap / EMA target | Redundancy-reduction | Self-distillation |
| **Negatives needed?** | No | No | No |
| **Collapse prevention** | Predictor + EMA asymmetry | Cross-correlation (λ=0.0051) | Centering (mom=0.9) + sharp τ |
| **Projector output dim** | 256-d | 2048-d | 65 536-d |
| **Backbone init** | `pretrained=False` | `pretrained=False` | `pretrained=True` ⚠ |
| **Augmentation** | Asymmetric (blur p=1.0 / p=0.1) | Symmetric, gentle (CJ 0.2) | Multi-crop: 2×224 + 4×96 |
| **Crop scale** | 0.08 – 1.0 | 0.40 – 1.0 | 0.40–1.0 (global) / 0.05–0.4 (local) |
| **Optimizer** | AdamW lr=3e-4 wd=1.5e-6 | AdamW lr=3e-4 wd=1e-4 | AdamW 5e-5 (enc) + 3e-4 (head) |
| **LR schedule** | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| **Grad clipping** | None | None | max_norm=3.0 |
| **GFLOPs (encoder)** | 0.394 | 0.394 | 0.394 |
| **Effective views/step** | 2 | 2 | 6 (2 global + 4 local) |

> ⚠ DINO uses `pretrained=True` (ImageNet init for both student and teacher). BYOL and Barlow Twins use `pretrained=False`. This gives DINO a prior advantage in short (30-epoch) runs. For strictly fair comparison set `pretrained=False` in DINO's Cell 6.

---

## Results Comparison Table

| Metric | BYOL | Barlow Twins | DINO | Notes |
|---|:---:|:---:|:---:|---|
| **Linear probe accuracy** | 0.3012 | 0.2986 | 0.4550 | LogReg C=1, frozen features |
| **Macro precision** | 0.2473 | 0.2653 | 0.4652 | |
| **Macro recall** | 0.2531 | 0.2473 | 0.4883 | |
| **Macro F1** | 0.2424 | 0.2409 | 0.4753 | equal class weight |
| **Macro ROC-AUC** | 0.6479 | 0.6348 | 0.7902 | OvR |
| **Silhouette score** | -0.1544 | -0.0459 | -0.0305 | test embeddings, max 2000 pts |
| **Fine-tune accuracy** | 0.4505 | 0.4332 | 0.6836 | 20 ep, LR=1e-4 |
| **Fine-tune F1** | 0.4607 | 0.4500 | 0.7204 | |
| **LP → FT delta (Δ acc)** | +0.1493 | +0.1346 | +0.2286 | positive = FT improves LP |
| **kNN-1 accuracy** | 0.3226 | 0.2531 | 0.5798 | cosine, frozen emb |
| **kNN-5 accuracy** | 0.3044 | 0.2647 | 0.5463 | |
| **kNN-20 accuracy** | 0.2955 | 0.2620 | 0.5250 | |
| **Label efficiency 1%** | 0.2447 | 0.2094 | 0.3017 | LP on 1% labeled data |
| **Label efficiency 5%** | 0.2723 | 0.2678 | 0.3391 | |
| **Label efficiency 10%** | 0.2727 | 0.2839 | 0.3596 | |
| **Label efficiency 25%** | 0.2919 | 0.2857 | 0.3730 | |
| **Label efficiency 50%** | 0.2972 | 0.2812 | 0.4113 | |
| **Pretrain time (s)** | 19399 | 11545 | 19452 | wall-clock |
| **Fine-tune time (s)** | 2138 | 1879 | 1826 | |

---

## Shallow-Head Breakdown (frozen features)

| Head | BYOL acc | Barlow Twins acc | DINO acc |
|---|:---:|:---:|:---:|
| Linear Probe | 0.3012 | 0.2986 | 0.4550 |
| MLP (256→128) | 0.3160 | 0.2692 | 0.5352 |
| SVM RBF (C=10) | 0.2723 | 0.2785 | 0.5441 |
| Decision Tree (depth 12) | 0.2865 | 0.2687 | 0.3039 |
| Random Forest (100 trees) | 0.3587 | 0.2901 | 0.4091 |

---

## Output Files per Notebook

| File | Contents |
|---|---|
| `pretrain_loss_{SSL}.png` | Loss curve over 30 pretrain epochs |
| `finetune_curves_{SSL}.png` | Fine-tune loss + train accuracy (2 panels) |
| `heads_le_{SSL}.png` | Bar chart (heads) + label-efficiency curve |
| `metrics_{SSL}.png` | Confusion matrix + per-class ROC curves |
| `embeddings_{SSL}.png` | PCA / t-SNE / UMAP side-by-side (N≤2000) |
| `ssl_summary_{SSL}.csv` | All scalar metrics in one row |
| `heads_{SSL}.csv` | Per-head: acc, F1, AUC, train_t, test_t |
| `label_eff_{SSL}.csv` | Label-efficiency: pct, n_samples, acc |
| `{method}_encoder.pt` | Saved model weights |

---

## Design Decisions & Rationale

**Why EfficientNet-B0?** Lightweight (0.394 GFLOPs), fast to pretrain, still produces rich 1280-d features. Keeps GPU memory within Kaggle's 16 GB limit even with DINO's 6-view multi-crop.

**Why gentle augmentation for Barlow Twins?** PM2.5 haze severity is encoded in global sky colour and atmospheric scattering gradients. Aggressive ColorJitter (0.4) can destroy this signal. Barlow Twins uses CJ=0.2 and large crop scale (0.40) to preserve it.

**Why does DINO use batch=16?** Each image produces 6 views (2×224px global + 4×96px local). At batch=16 the actual number of forward passes per step is 96 — comparable to BYOL/BarlowTwins at batch=32 with 2 views (64 passes), while staying within GPU memory.

**Why pool train + test for pretraining?** SSL is unsupervised — no labels are used during pretraining. Pooling all available images is standard practice and increases diversity. Evaluation uses a fresh 80/20 stratified split of the pool, so test-set images appear in train features only, not in the evaluation split used for metrics.
