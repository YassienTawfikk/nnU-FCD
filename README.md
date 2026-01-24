# Evaluation of nnU-Net for FCD II Lesions Segmentation in FLAIR MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: OpenNeuro](https://img.shields.io/badge/Dataset-OpenNeuro%20ds004199-blue.svg)](https://openneuro.org/datasets/ds004199)

**Official PyTorch Implementation** of the paper: *Evaluation of nnU-Net for FCD II Lesion Segmentation in FLAIR MRI*.

<p align="center">
  <img src="figures/nnU-Net PIPELINE.png" alt="nnU-Net Pipeline" width="100%">
  <br>
  <em>Figure: Overview of the proposed nnU-Net v2 pipeline.</em>
</p>

## Abstract

Epilepsy is one of the most common neurological disorders worldwide. Focal Cortical Dysplasia (FCD) is a major cause of drug-resistant epilepsy, often presenting as subtle lesions on MRI scans. While deep learning shows promise for automated FCD segmentation, existing methods achieved limited performance and used inconsistent dataset splits that prevent reproducible comparison. In this study, we propose an nnU-Net-based approach for automated FCD segmentation using the Bonn FCD Type II dataset. To improve model performance, our proposed approach employs oversampling of underrepresented FCD Type II radiological features and uses extensive augmentation during the training phase. We achieved an average validation Pseudo Dice of 0.56 compared to 0.45 in prior work, demonstrating significant performance improvement. This work provides the first complete evaluation on the dataset's standard train/test split, reporting both validation and test set performance to establish a reproducible benchmark for future method comparison. By providing both improved segmentation performance and a standardized evaluation framework, our work advances automated FCD segmentation toward better presurgical evaluation and improved outcomes for patients with drug-resistant epilepsy.

## Key Contributions

- **Reproducible Benchmark**: First complete evaluation on the Bonn FCD Type II dataset's standard train/test split, establishing a reliable baseline for future comparisons.
- **Improved Performance**: Achieved significant improvement in segmentation performance (Validation Pseudo Dice 0.56 vs. 0.45 in prior work) through radiological feature-based oversampling and extensive augmentation.
- **Radiological Feature-Awareness**: Addressed class imbalance by oversampling subtle and underrepresented radiological features (Transmantle Sign, Gray-White Matter Blurring).

## Methodology

### Pipeline Overview



- **Architecture**: nnU-Net (3d_fullres)
- **Data Splitting**: Adherence to the standard 57/28 train/test split to ensure reproducibility.
- **Preprocessing**: nnU-Net v2 automated preprocessing with resampling.
- **Radiological Feature-Based Sampling**: 3x oversampling for subjects with rare radiological features (Transmantle Sign, Gray-White Matter Blurring) to address class imbalance.
- **Extensive Augmentation**: Enhanced augmentation parameters to improve generalization:
    - **Rotation**: ±60°
    - **Scaling**: [0.70, 1.50]
    - **Brightness/Contrast**: [0.5, 1.5]
    - **Gaussian Noise**: $\sigma=0.2$
    - **Gamma**: [0.4, 2.0]

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

The workflow is organized into sequential notebooks for reproducibility:

1. **Preprocessing**: `notebooks/00_Data_Preprocessing.ipynb`
   - Prepares the dataset for nnU-Net.
2. **Training**: `notebooks/01_Train_Oversampling.ipynb`
   - Trains the model with radiological feature-based oversampling.
3. **Inference**: `notebooks/03_Inference.ipynb`
   - Runs inference on the test set and generates evaluation metrics.

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
@article{Tawfik2024FCD,
  title={Evaluation of nnU-Net for FCD II Lesion Segmentation in FLAIR MRI},
  author={Tawfik, Yassien and Marwan, Mazen and Yasser, Mohamed and Mahmoud, Nancy and Mosaad, Madonna and Salman, Mahmoud and Basha, Tamer and Khalaf, Aya},
  journal={Department of Systems and Biomedical Engineering, Cairo University},
  year={2024}
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
