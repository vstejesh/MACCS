# MACCS: mRNA-Anchored Cross-Attention Classifier for Cervical Cancer Subtyping

Binary classification of cervical cancer subtypes (SCC vs ADC) using a Star Topology cross-attention network over four multi-omics modalities. This repo implements **Stage 3** of the full pipeline described here — it takes a preprocessed, imputed `.npz` file as input and handles the rest.

---

## Pipeline Overview

The full pipeline has three stages. This repo covers only **Stage 3**:

```
[Stage 1] MIMIR Imputation       → not in this repo (see MIMIR GitHub below)
[Stage 2] mRMR Feature Selection → not in this repo (described below)
[Stage 3] Cross-Attention Model  → THIS REPO
```

---

## Data Preparation (Stages 1 & 2)

A ready-to-use `.npz` file is already included in the repo (`imputed_combined_omics_383.npz`). You only need to follow the steps below if you want to reproduce the data from scratch.

### Stage 1 — MIMIR Imputation

MIMIR is a masked autoencoder pretrained on TCGA pan-cancer data that reconstructs missing omics and gene-level values. Use the official repo and its pretrained weights:

> **[Noble-Lab/MIMIR](https://github.com/Noble-Lab/MIMIR)**

Run MIMIR using the exact feature vocabulary it was pretrained on (mRNA: 3,007 genes; CNV: 3,106; Methylation: 3,139 CpG sites; miRNA: 383). This produces a folder of imputed omic files per sample.

**Note for HTMCP samples:** Before running MIMIR, HTMCP mRNA/miRNA must be re-aligned to TCGA's reference space using the variance-stabilizing transform described in the paper. The Drive folder linked below contains already-normalized HTMCP data.

**Google Drive — raw (pre-MIMIR) and imputed (post-MIMIR) omic folders:**
> [https://drive.google.com/drive/folders/1wPoyP7TffIwZ1he7b-19NaaAqWxLFju3?usp=drive_link](https://drive.google.com/drive/folders/1wPoyP7TffIwZ1he7b-19NaaAqWxLFju3?usp=drive_link)

### Stage 2 — mRMR Feature Selection + NPZ Creation

After imputation, reduce each modality from ~3k features down to 383 using mRMR (max-relevance, min-redundancy), then pack everything into a single `.npz`. This step is **not included in this repo**, but the resulting file is:

```
imputed_combined_omics_383.npz
```

To verify: running mRMR on the post-MIMIR folder from the Drive link and converting to `.npz` reproduces the exact same file already in the repo.

The `.npz` has the following keys:
```
mRNA         → (441, 383) float32
Methylation  → (441, 383) float32
CNV          → (441, 383) float32
miRNA        → (441, 383) float32
y            → (441,)     int   — 0=ADC, 1=SCC
```

---

## Installation

```bash
git clone https://github.com/<your-username>/mRNA-fusion-CC-Classifier.git
cd mRNA-fusion-CC-Classifier
pip install -r requirements.txt
```

---

## Usage

```bash
python train.py
```

All configuration is at the top of `train.py`:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `imputed_combined_omics_383.npz` | Path to the input npz |
| `CENTRAL_OMIC` | `"mRNA"` | Master modality for cross-attention |
| `NUM_FOLDS` | `5` | Stratified K-Fold splits |
| `NUM_EPOCHS` | `10` | Epochs per fold |
| `BATCH_SIZE` | `16` | |
| `LEARNING_RATE` | `1e-4` | Adam optimizer |

To test a different master modality, change `CENTRAL_OMIC` to one of `"mRNA"`, `"Methylation"`, `"CNV"`, or `"miRNA"`.

---

## Outputs

After training, the following files are saved to the working directory:

```
best_kfold_model.pth        # Best fold model weights
confusion_matrix_cv.png     # Aggregated OOF confusion matrix
roc_curve_cv.png            # Aggregated OOF ROC curve
latent_space_pca.png        # PCA of the 64-dim latent space
initial_weights.pth         # Shared starting weights (used internally across folds)
```

Per-fold and aggregated metrics (Macro F1, Weighted F1, ROC-AUC, Sensitivity, Specificity) are printed to stdout.

---

## Repository Structure

```
mRNA-fusion-CC-Classifier/
├── model.py                        # Star Topology cross-attention model
├── dataset.py                      # Dataset class + SMOTE integration
├── train.py                        # 5-fold CV training + evaluation
├── imputed_combined_omics_383.npz  # Preprocessed data (included)
└── requirements.txt
```

---

## Model Architecture
<img width="1343" height="706" alt="image" src="https://github.com/user-attachments/assets/ad1a64c7-c690-46da-a3cf-fbb2913af86d" />


`model.py` implements a three-part network:

**`MicroEncoder`** — a single linear projection (no hidden layers) with BatchNorm and Dropout (p=0.4) that maps each modality's 383 features to a 16-dimensional embedding.

**`MicroCrossAttention`** — single-head attention where the slave modality embedding is the Query and the master (mRNA) is the Key/Value. One attention block per non-central modality.

**`GeneralizedCrossAttn`** — the full model. Encodes all four modalities, applies cross-attention from each slave into the master, concatenates the four 16-dim vectors (64-dim total), and classifies with a linear → BN → ReLU → Dropout → linear head.

The `central_omic` argument is configurable — any of the four modalities can be set as master.

---

## Reference

If you use this code, please also cite the MIMIR paper:

> Nambiar et al., *Unified imputation of missing data modalities and features in multi-omic data via shared representation learning*, bioRxiv 2026. [https://doi.org/10.64898/2026.02.04.703630](https://doi.org/10.64898/2026.02.04.703630)
