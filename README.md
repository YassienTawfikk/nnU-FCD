# Benchmarking FCD Lesion Detection: A Leakage-Free and Clinically Realistic Evaluation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: OpenNeuro](https://img.shields.io/badge/Dataset-OpenNeuro%20ds004199-blue.svg)](https://openneuro.org/datasets/ds004199)

**Official PyTorch Implementation** of the paper: *Benchmarking FCD Lesion Detection: A Leakage-Free and Clinically Realistic Evaluation Framework*.

## Abstract

Epilepsy affects more than 60 million people worldwide, with approximately one-third experiencing drug-resistant epilepsy. Focal Cortical Dysplasia (FCD) is a major cause in these patients, frequently presenting as MRI-negative lesions that evade visual detection. Surgical resection offers potential seizure freedom, but success critically depends on accurate lesion localization. Current deep learning approaches report high segmentation performance; however, these metrics may be inflated due to evaluation protocols that permit data leakage and lack rigorous generalization testing.

This study establishes a **leakage-free benchmark** for automated FCD lesion segmentation using the histologically confirmed Bonn FCD Type II dataset. We enforce strict subject-level cross-validation to prevent slice-level data leakage and evaluate model performance without heuristic post-processing. Our **data-centric optimization strategy** prioritizes preservation of cortical boundaries and careful handling of rare, subtle lesion phenotypes (like Transmantle Sign and Gray-White Matter Blurring) often missed by standard architectures.

## Key Contributions

- **Leakage-Free Benchmark**: Strict subject-level cross-validation on the Bonn FCD Type II dataset to prevent slice-level data leakage.
- **Clinically Realistic Evaluation**: Evaluation on full 3D volumes without skull-stripping or heuristic post-processing (CCA), reflecting real-world clinical performance.
- **Phenotype-Aware Oversampling**: A novel data-centric strategy that prioritizes rare radiological signs (Transmantle Sign, Gray-White Matter Blurring) during training.
- **Transparent Baseline**: We expose the "Leakage Gap" in existing literature, where reported Dice scores (~0.45) are inflated by improper validation.

## Results

Our rigorous evaluation establishes a realistic baseline for FCD detection, distinguishing true clinical utility from inflated metrics.

| Method | Validation Dice | Test Dice (Strict) | Notes |
| :--- | :---: | :---: | :--- |
| Baseline (Standard nnU-Net) | 0.45 ± 0.05 | -- | High variance, erratic convergence |
| **Proposed Framework** | **0.56 ± 0.18** | **0.23 ± 0.03** | **Stable convergence, captures rare phenotypes** |

*Table: Comparative benchmarking. Note that "Test Dice (Strict)" is lower than typical literature values (e.g., 0.41-0.45) because we prevent data leakage and do not use post-processing.*

## Methodology

### Pipeline Overview

- **Architecture**: nnU-Net (3d_fullres)

- **Preprocessing**: No skull-stripping (preserves cortical boundaries).
- **Augmentation**: Aggressive spatial (±60° rotation, 0.7-1.5x scaling) and intensity transformations.
- **Sampling**: Custom `Abnormality-Aware Sampling` to oversample subjects with rare FCD features (3x frequency).
- **Leakage Prevention**: Validation and Test sets are strictly isolated at the subject level.

### Repository Structure

```
.
├── notebooks/           # Jupyter Notebooks for Data Prep, Training, and Inference
├── src/                 # Source code, Configuration, and Custom nnU-Net Modules
├── data/                # Dataset directory (Bonn FCD II)
├── figures/             # Generated figures and plots
├── results/             # Evaluation metrics and checkpoints
└── requirements.txt     # Python dependencies
```

## Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Setup

Download the [Bonn FCD II dataset](https://openneuro.org/datasets/ds004199) and place it in the `data/` directory. Ensure `participants.tsv` is in `data/participants-data/`.

### 3. Running the Pipeline

The workflow is organized into sequential notebooks:

1. **Preprocessing**: `notebooks/00_Data_Preprocessing.ipynb`
2. **Training**: `notebooks/01_Train_Oversampling.ipynb`
3. **Inference**: `notebooks/03_Inference.ipynb`

## Citation

If you use this code or dataset split in your research, please cite our paper:

```bibtex
@article{Tawfik2024FCD,
  title={Benchmarking FCD Lesion Detection: A Leakage-Free and Clinically Realistic Evaluation Framework},
  author={Tawfik, Yassien and Marwan, Mazen and Yasser, Mohamed and Mahmoud, Nancy and Mosaad, Madonna and Khalaf, Aya and Yasser, Mahmoud},
  journal={Department of Systems and Biomedical Engineering, Cairo University},
  year={2024}
}
```

## Authors

- **Yassien Tawfik**, Mazen Marwan, Mohamed Yasser, Nancy Mahmoud, Madonna Mosaad (Cairo University)
- **Aya Khalaf** (Yale University)
- **Mahmoud Yasser** (Western University)
