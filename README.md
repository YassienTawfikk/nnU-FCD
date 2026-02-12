# Evaluation of nnU-Net for FCD II Lesions Segmentation in FLAIR MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: OpenNeuro](https://img.shields.io/badge/Dataset-OpenNeuro%20ds004199-blue.svg)](https://openneuro.org/datasets/ds004199)

**Official PyTorch Implementation** of the paper: *Evaluation of nnU-Net for FCD II Lesion Segmentation in FLAIR MRI*.

<p align="center">
  <img src="figures/nnU-Net PIPELINE.png" alt="nnU-Net Pipeline" width="100%">
  <br>
  <em>Figure: Overview of the proposed nnU-Net v2 pipeline.</em>
</p>

## Pretrained Models (Checkpoints)

We provide pretrained nnU-Net v2 checkpoints for the experiment.

- **Recommended (Proposed: BSL + OVS + Aug)** — best overall performance  
  **Google Drive:** [Download checkpoint](https://drive.google.com/file/d/1KL6BwuhixWB1SuQzru_pkD2Jx4_8yjQY/view?usp=share_link)
  **Notes:** trained on the standard Bonn FCD II split used in this repo.

## Abstract

Epilepsy is one of the most common neurological disorders worldwide. Focal Cortical Dysplasia (FCD) is a major cause of drug-resistant epilepsy, often presenting as subtle lesions on MRI scans. While deep learning shows promise for automated FCD segmentation, existing methods achieved limited performance and used inconsistent dataset splits that prevent reproducible comparison. In this study, we propose an nnU-Net-based approach for automated FCD segmentation using the Bonn FCD Type II dataset. To improve model performance, our proposed approach employs oversampling of underrepresented FCD Type II radiological features and uses extensive augmentation during the training phase. We achieved an average validation Pseudo Dice of 0.56 compared to 0.45 in prior work, demonstrating significant performance improvement. This work provides the first complete evaluation on the dataset's standard train/test split, reporting both validation and test set performance to establish a reproducible benchmark for future method comparison. By providing both improved segmentation performance and a standardized evaluation framework, our work advances automated FCD segmentation toward better presurgical evaluation and improved outcomes for patients with drug-resistant epilepsy.

## Key Contributions

- **Reproducible Benchmark**: First complete evaluation on the Bonn FCD Type II dataset's standard train/test split, establishing a reliable baseline for future comparisons.
- **Improved Performance**: Achieved significant improvement in segmentation performance (Validation Pseudo Dice 0.56 vs. 0.45 in prior work) through radiological feature-based oversampling and extensive augmentation.
- **Radiological Feature-Awareness**: Addressed class imbalance by oversampling subtle and underrepresented radiological features (Transmantle Sign, Gray-White Matter Blurring).

## Methodology

### 1. Addressing Class Imbalance
The Bonn FCD Type II dataset exhibits significant imbalance in radiological phenotypes, with subtle features being underrepresented.

<p align="center">
  <img src="figures/Radiological_Abnormality_Distribution.png" alt="Radiological Abnormality Distribution" width="80%">
  <br>
  <em>Figure: Distribution of radiological abnormalities in the dataset. Note the scarcity of "Transmantle sign" and "Gray-White Matter Blurring".</em>
</p>

### 2. Radiological Feature-Based Oversampling
To address this, we implemented a custom `nnUNetDataLoader` that increases the sampling probability for subjects exhibiting these rare features by a factor of 3.

**Oversampled Subjects:**
> `sub-00001`, `sub-00003`, `sub-00014`, `sub-00015`, `sub-00016`, `sub-00018`, `sub-00024`, `sub-00027`, `sub-00033`, `sub-00038`, `sub-00040`, `sub-00044`, `sub-00050`, `sub-00053`, `sub-00055`, `sub-00058`, `sub-00060`, `sub-00063`, `sub-00065`, `sub-00073`, `sub-00077`, `sub-00078`, `sub-00080`, `sub-00081`, `sub-00083`, `sub-00087`, `sub-00089`, `sub-00097`, `sub-00098`, `sub-00101`, `sub-00105`, `sub-00109`, `sub-00112`, `sub-00115`, `sub-00116`, `sub-00122`, `sub-00123`, `sub-00126`, `sub-00130`, `sub-00132`, `sub-00133`, `sub-00138`, `sub-00146`

**Implementation Snippet:**
```python
# From src/nnunet_extensions/custom_dataloader.py

# Increase weight for rare subjects
for idx, case_id in enumerate(case_ids):
    subject_id = str(case_id).split('_')[0]
    
    if subject_id in self.rare_subjects:
        # Increase probability (OVERSAMPLE_FACTOR = 3.0)
        sampling_probs[idx] *= self.oversample_factor 

# Normalize probabilities
sampling_probs = sampling_probs / sampling_probs.sum()
self.sampling_probabilities = sampling_probs
```

### 3. Extensive Data Augmentation
We significantly increased the intensity and range of data augmentations in `nnUNetPlans.json` to improve generalization on this small dataset.

**Configuration Snippet:**
```python
# Doubled spatial augmentations
plans["data_augmentation"]["spatial"]["rotation"] = {
    "x": 60 * (math.pi / 180), # ±60 degrees
    "y": 60 * (math.pi / 180),
    "z": 60 * (math.pi / 180)
}
plans["data_augmentation"]["spatial"]["scale_range"] = [0.70, 1.50]

# Doubled intensity augmentations
plans["data_augmentation"]["intensity"]["brightness"] = [0.5, 1.5]
plans["data_augmentation"]["intensity"]["contrast"] = [0.5, 1.5]
plans["data_augmentation"]["intensity"]["gaussian_noise_std"] = 0.2
plans["data_augmentation"]["intensity"]["gamma_range"] = [0.4, 2.0]
```

### Repository Structure

```
.
├── notebooks/
│   ├── 00_Dataset_Preparation.ipynb      # Common dataset setup
│   ├── proposed_method/                  # Proposed Method (OVS + Aug)
│   ├── baseline_oversampling_method/     # Ablation: Baseline + Oversampling
│   └── baseline_method/                  # Ablation: Baseline
├── src/
│   ├── config.py
│   └── nnunet_extensions/                # Shared custom Trainer & DataLoader
├── data/                # Dataset directory (Bonn FCD II)
├── figures/             # Generated figures and plots
├── results/             # Evaluation metrics and checkpoints
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- nnU-Net v2

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YassienTawfikk/nnU-FCD.git
   cd nnU-FCD
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

1. Download the [Bonn FCD II dataset](https://openneuro.org/datasets/ds004199).
2. Place the dataset in the `data/` directory.
3. Ensure `participants.tsv` is in `data/participants-data/`.

## Usage

The workflow is organized into separate experiments for reproducibility:

### Step 0: Data Setup
- Run `notebooks/00_Dataset_Preparation.ipynb` to download and preprocess the dataset.

### Experiment 1: Proposed Method (Best Performance)
The core contribution of the paper (Oversampling + Augmentation).
1. `notebooks/proposed_method/01_Train_Proposed_Model.ipynb`
2. `notebooks/proposed_method/03_Inference_Proposed.ipynb`

### Experiment 2: Baseline + Oversampling (Ablation)
Isolates the effect of radiological oversampling without extensive augmentation.
1. `notebooks/baseline_oversampling_method/01_Train_BSL_OVS_Model.ipynb`
2. `notebooks/baseline_oversampling_method/03_Inference_BSL_OVS.ipynb`

### Experiment 3: Baseline (Standard nnU-Net)
Standard nnU-Net training for performance comparison.
1. `notebooks/baseline_method/01_Train_Baseline_Model.ipynb`
2. `notebooks/baseline_method/03_Inference_Baseline.ipynb`

## Results

We evaluated our model on the standard train/test split.

### Ablation Study (Test Set)

Our ablation study demonstrates the progressive improvement of our contributions:

| Experiment | Best Val. PDS | Best Test DS |
| :--- | :---: | :---: |
| Baseline (BSL) | 0.545 | 0.168 |
| BSL + Oversampling (OVS) | 0.466 | 0.219 |
| **BSL + OVS + Augmentation (Proposed)** | **0.667** | **0.256** |

<p align="center">
  <img src="figures/Ablation-Study.png" alt="Ablation Study Results" width="100%">
  <br>
  <em>Figure: Qualitative improvements from Ablation Study. Row 1: Baseline. Row 2: +Oversampling. Row 3: +Augmentation (Proposed).</em>
</p>

### Comparison with State-of-the-Art (Standard Split)

| Method | Post-Processing | Mean Val. PDS | Best Val. PDS |
| :--- | :---: | :---: | :---: |
| Joshi et al. | CCA | 0.45 | 0.52 |
| **Proposed** | **None** | **0.54** | **0.65** |

*Note: Validation PDS (Pseudo Dice Score) is computed patch-wise during training/validation.*

## Citation

If you use this code or dataset split in your research, please cite our paper:

```bibtex
@article{Tawfik2026FCD,
  title={Evaluation of nnU-Net for FCD II Lesion Segmentation in FLAIR MRI},
  author={Tawfik, Yassien and Marwan, Mazen and Yasser, Mohamed and Mahmoud, Nancy and Mosaad, Madonna and Salman, Mahmoud and Basha, Tamer and Khalaf, Aya},
  journal={Department of Systems and Biomedical Engineering, Cairo University},
  year={2026}
}
```

## Authors

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://medicine.yale.edu/profile/aya-khalaf/" target="_blank">
          <img src="https://ysm-res.cloudinary.com/image/upload/c_crop,x_354,y_0,w_2396,h_2400/c_fill,f_auto,q_auto:eco,dpr_2,w_650/v1/yms/prod/4426ecb7-0a4d-4e6d-af35-c329e1ae6e54" width="190px;" alt="Aya Khalaf"/>
          <br/><br/>
          <sub><b>Dr. Aya Khalaf</b></sub>
        </a>
        <br/>
        <sub>Yale University</sub>
      </td>
    </tr>
  </table>
</div>
<br/>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/YassienTawfikk" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="140px;" alt="Yassien Tawfik"/>
          <br/>
          <sub><b>Yassien Tawfik</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
      <td align="center">
        <a href="https://github.com/mohamedddyasserr" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/126451832?v=4" width="140px;" alt="Mohamed Yasser"/>
          <br/>
          <sub><b>Mohamed Yasser</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
      <td align="center">
        <a href="https://github.com/nancymahmoud1" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/125357872?v=4" width="140px;" alt="Nancy Mahmoud"/>
          <br/>
          <sub><b>Nancy Mahmoud</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
      <td align="center">
        <a href="https://github.com/Mazenmarwan023" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="140px;" alt="Mazen Marwan"/>
          <br/>
          <sub><b>Mazen Marwan</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>      
      <td align="center">
        <a href="https://github.com/madonna-mosaad" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/127048836?v=4" width="140px;" alt="Madonna Mosaad"/>
          <br/>
          <sub><b>Madonna Mosaad</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
    </tr>
  </table>
</div>
<br/>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/mahmoud1yaser" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/88372358?v=4" width="165px;" alt="Mahmoud Salman"/>
          <br/>
          <sub><b>Mahmoud Salman</b></sub>
        </a>
        <br/>
        <sub>Western University</sub>
      </td>
    </tr>
  </table>
</div>
<br/>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://www.linkedin.com/in/tamer-basha-b81812ab/" target="_blank">
          <img src="https://media.licdn.com/dms/image/v2/C5103AQEkkCY9JaaHTQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1517586051833?e=1770854400&v=beta&t=dZaKaGmqW-0RJwHGhjnH9RgE19UAbu9fow0PsU43mFU" width="135px;" alt="Tamer Basha"/>
          <br/>
          <sub><b>Tamer Basha</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
    </tr>
  </table>
</div>
