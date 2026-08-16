# Unified Anomaly Utilization Framework for Industrial Visual Defect Detection and Localization

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20365977.svg)](https://doi.org/10.5281/zenodo.20365977)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **CAUNet**, a unified anomaly utilization framework for industrial visual defect detection and localization.

> **Paper:** *Unified Anomaly Utilization Framework for Industrial Visual Defect Detection and Localization*
> **Journal:** *The Visual Computer*
> **Code Repository:** https://github.com/xl-li-5913/CAUNet
> **Software Archive DOI:** [10.5281/zenodo.20365977](https://doi.org/10.5281/zenodo.20365977)
> **Journal Article DOI:** To be added after publication.

> **Note:** The Zenodo DOI above refers to the archived **software release** of CAUNet and is distinct from the DOI of the journal article.

---

## Overview

Industrial anomaly detection plays an important role in automated quality inspection and manufacturing safety. However, real industrial anomalies are scarce and difficult to annotate, which limits the direct application of supervised learning methods in practical industrial scenarios.

Recent pseudo-anomaly-based methods alleviate this problem by constructing synthetic anomalies from normal samples. Nevertheless, pseudo anomalies are often used only as local perturbations, and anomaly information is not sufficiently transferred and consolidated during feature learning.

To address these limitations, we propose **CAUNet**, a unified anomaly utilization framework that progressively organizes anomaly information through:

* **structure-aware anomaly generation**,
* **anomaly knowledge transfer**, and
* **adaptive anomaly response enhancement**.

CAUNet models anomaly-related information in both the data space and feature space to improve anomaly representation learning and defect localization.

---

## Framework Architecture

CAUNet mainly consists of three modules:

### 1. Fuzzy-guided Pseudo Anomaly Generation (FPAG)

FPAG constructs structure-aware pseudo anomalies through fuzzy-guided anomaly injection.

Main characteristics:

* Models anomaly degree using pixel-level fuzzy memberships
* Introduces structural guidance into pseudo-anomaly generation
* Preserves spatial continuity of generated anomalous regions
* Provides anomaly information for subsequent feature learning

### 2. Statistics-guided Anomaly Knowledge Transfer (SAKT)

SAKT transfers anomaly-related information from pseudo-anomalous samples into the feature representation space.

Main characteristics:

* Models anomaly-related feature statistics
* Dynamically modulates feature channels
* Enhances anomaly-aware feature representation
* Facilitates anomaly knowledge transfer and consolidation

### 3. Position-adaptive Dynamic Anomaly Amplification (PADAA)

PADAA adaptively enhances anomaly responses at different spatial positions.

Main characteristics:

* Dynamically enhances local anomaly responses
* Adapts to spatially varying anomaly patterns
* Improves sensitivity to fine-grained defects
* Strengthens defect localization capability

---

## Experimental Results

### MVTec AD

The results reported in the revised manuscript are obtained from three independent runs and are presented as mean ± standard deviation.

| Metric            |       CAUNet |
| ----------------- | -----------: |
| Image-level AUROC | 99.36 ± 0.01 |
| Pixel-level AUROC | 98.25 ± 0.04 |
| AUPRO             | 94.98 ± 0.05 |

CAUNet also shows competitive performance on the **VisA** and **MPDD** industrial anomaly detection benchmarks. Please refer to the manuscript for the complete category-wise results and comparisons with existing methods.

---

## Repository Structure

```text
CAUNet/
├── ablation/                         # Ablation experiment implementations
│   ├── main_nofpag.py
│   ├── main_nofpagandpadaa.py
│   ├── main_nofpagandsakt.py
│   ├── main_nofpagandsaktandpadaa.py
│   ├── main_nopadaa.py
│   ├── main_nosakt.py
│   └── main_nosaktandpadaa.py
│
├── data/                             # Dataset directory
├── dataset/                          # Dataset loading and pseudo-anomaly generation
├── model/                            # Network architectures
├── utils/                            # Utility functions
├── inference.py                      # Inference and evaluation script
├── train_CAUNet.py                   # CAUNet training script
├── requirements.txt                  # Python dependencies
├── LICENSE                           # Software license
└── README.md                         # Project documentation
```

---

## Environment

The experiments were conducted using the following main software environment:

* **Python:** 3.9+
* **PyTorch:** 2.3.1
* **Torchvision:** 0.18.1
* **CUDA:** 11.8
* **cuDNN:** 8.7.0
* **Precision:** FP32

The main Python dependencies are specified in `requirements.txt`.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

The released `requirements.txt` uses the CUDA 11.8 build of PyTorch 2.3.1.

---

## Dataset Preparation

The datasets used in this work are publicly available and are **not redistributed** in this repository. Please download them from their official sources and follow the corresponding licenses and terms of use.

### MVTec AD

Download:

https://www.mvtec.com/company/research/datasets/mvtec-ad

Place the dataset under:

```text
data/mvtec/
```

Example directory structure:

```text
data/
└── mvtec/
    ├── bottle/
    ├── cable/
    ├── capsule/
    ├── carpet/
    ├── grid/
    └── ...
```

### VisA

Official repository:

https://github.com/amazon-science/spot-diff

Place the prepared dataset under the corresponding dataset directory used by the training/evaluation scripts.

### MPDD

Official repository:

https://github.com/stepanje/MPDD

Place the prepared dataset under the corresponding dataset directory used by the training/evaluation scripts.

Before running an experiment, please verify that the dataset path in the training or inference script is consistent with the local dataset location.

---

## Training

Train CAUNet using:

```bash
python train_CAUNet.py
```

The training script contains the model construction, optimization, pseudo-anomaly generation, and checkpoint-saving procedure used for CAUNet.

Please ensure that:

1. the dataset has been downloaded and placed in the correct directory;
2. all Python dependencies have been installed;
3. the dataset path and target category are configured correctly; and
4. the output directory is writable.

---

## Inference and Evaluation

Evaluate a trained CAUNet model using:

```bash
python inference.py
```

The inference script performs anomaly detection/localization and reports the evaluation metrics used in the manuscript.

---

## Evaluation Metrics

The main evaluation metrics are:

* **Image-level AUROC (I-AUROC)**
* **Pixel-level AUROC (P-AUROC)**
* **Area Under the Per-Region Overlap Curve (AUPRO/PRO)**

These metrics are used to evaluate image-level anomaly detection and pixel-level anomaly localization performance.

---

## Visualization

The inference pipeline supports visualization of anomaly localization results, including:

* input images,
* ground-truth masks,
* predicted anomaly maps, and
* localization/segmentation results.

These visualizations can be used to qualitatively inspect anomaly responses and localization behavior.

---

## Ablation Experiments

The `ablation/` directory contains the implementations used to evaluate the contributions of FPAG, SAKT, and PADAA.

| Script                          | Configuration                        |
| ------------------------------- | ------------------------------------ |
| `main_nofpag.py`                | CAUNet without FPAG                  |
| `main_nosakt.py`                | CAUNet without SAKT                  |
| `main_nopadaa.py`               | CAUNet without PADAA                 |
| `main_nofpagandsakt.py`         | CAUNet without FPAG and SAKT         |
| `main_nofpagandpadaa.py`        | CAUNet without FPAG and PADAA        |
| `main_nosaktandpadaa.py`        | CAUNet without SAKT and PADAA        |
| `main_nofpagandsaktandpadaa.py` | CAUNet without FPAG, SAKT, and PADAA |

The ablation implementations follow the same overall training and evaluation framework as the complete CAUNet model, with the corresponding component(s) removed.

---

## Reproducibility

The repository provides the source code required to train and evaluate CAUNet, together with the ablation implementations and dataset acquisition information.

For reproducibility, the random seed is fixed to **111**.

The implementation fixes the random states of PyTorch, CUDA, NumPy, and Python's built-in `random` module, and deterministic cuDNN behavior is enabled.

```python
torch.manual_seed(111)
torch.cuda.manual_seed_all(111)
np.random.seed(111)
random.seed(111)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

All reported CAUNet results were obtained from **three independent runs**, with the random seed fixed to **111 in each run**. The quantitative results in the manuscript are reported as **mean ± standard deviation** over the three runs.

To reproduce the CAUNet experiments:

1. Download and prepare the target dataset.
2. Install the dependencies from `requirements.txt`.
3. Verify the dataset path and experimental settings in the corresponding script.
4. Run `train_CAUNet.py`.
5. Run `inference.py` using the trained model.
6. Repeat the experiment three times using the fixed random seed of 111 when reproducing the reported repeated-run protocol.

Because hardware, CUDA kernels, and low-level library implementations may differ across systems, very small numerical differences may occur between environments.

---

## Code and Data Availability

The source code, training implementation, inference implementation, and ablation experiment code for CAUNet are publicly available at:

https://github.com/xl-li-5913/CAUNet

A permanent archived software release is available through Zenodo:

https://doi.org/10.5281/zenodo.20365977

The MVTec AD, VisA, and MPDD datasets are publicly available from their respective official sources listed in the **Dataset Preparation** section. The datasets themselves are not redistributed in this repository.

The Zenodo DOI identifies the archived **software release** associated with CAUNet and should not be confused with the DOI of the journal article.

---

## Citation

If you find this repository useful in your research, please cite the corresponding journal article after publication.

```bibtex
@article{CAUNet2026,
  title   = {Unified Anomaly Utilization Framework for Industrial Visual Defect Detection and Localization},
  author  = {CAUNet Authors},
  journal = {The Visual Computer},
  year    = {2026}
}
```

The complete bibliographic information and journal article DOI will be updated after publication.

For the archived software release, please use the Zenodo record:

```text
CAUNet software archive
DOI: 10.5281/zenodo.20365977
```

---

## License

This project is released under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

The MVTec AD, VisA, MPDD datasets, and any third-party components remain subject to their respective licenses and terms of use.

---

## Acknowledgements

CAUNet is developed within the reverse-distillation-based industrial anomaly detection framework. We thank the authors of related open-source industrial anomaly detection methods for making their implementations publicly available and supporting reproducible research.
