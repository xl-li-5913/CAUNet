# CAUNet: Unified Anomaly Utilization Framework for Industrial Visual Defect Detection and Localization

Official implementation of **CAUNet**, a unified anomaly utilization framework for industrial visual defect detection and localization.

> **Paper**: *Unified Anomaly Utilization Framework for Industrial Visual Defect Detection and Localization*  
> **Journal**: *The Visual Computer*

---

## Highlights

- Unified anomaly utilization framework for industrial anomaly detection
- Structure-aware pseudo anomaly generation
- Statistics-guided anomaly knowledge transfer
- Position-adaptive anomaly amplification
- Strong performance on MVTec AD, VisA, and MPDD
- Robust localization capability for fine-grained and irregular defects

---

## Overview

Industrial anomaly detection plays an important role in automated quality inspection and manufacturing safety. However, real industrial anomalies are scarce and difficult to annotate, which limits supervised learning methods in practical applications.

Recent pseudo-anomaly-based methods attempt to alleviate this issue by constructing synthetic anomalies on normal samples. Nevertheless, most existing approaches only use pseudo anomalies as local perturbation augmentation and fail to establish a complete anomaly utilization pipeline. As a result, anomaly information cannot be effectively transferred and consolidated during feature learning.

To address these limitations, we propose **CAUNet**, a unified anomaly utilization framework that systematically organizes:

- structure-aware anomaly generation,
- anomaly knowledge transfer,
- and adaptive anomaly response enhancement

into a progressive and reusable industrial defect inspection pipeline.

Unlike conventional pseudo-anomaly methods, CAUNet continuously models anomaly information in both the data space and feature space, enabling more discriminative anomaly representation learning and more accurate defect localization.

---

## Framework Architecture

The proposed CAUNet mainly consists of three key modules:

### 1. Fuzzy-guided Pseudo Anomaly Generation (FPAG)

FPAG constructs structure-aware pseudo anomalies through fuzzy-guided anomaly injection.

Main characteristics:

- Models anomaly degree using pixel-level fuzzy memberships
- Preserves structural continuity of industrial objects
- Reduces the distribution gap between synthetic anomalies and real defects
- Generates more realistic spatial defect patterns

---

### 2. Statistics-guided Anomaly Knowledge Transfer (SAKT)

SAKT transfers anomaly-related information from the data space into the feature representation space.

Main characteristics:

- Models anomaly distribution statistics
- Dynamically modulates feature channels
- Enhances anomaly-aware feature learning
- Improves anomaly knowledge propagation and consolidation

---

### 3. Position-adaptive Dynamic Anomaly Amplification (PADAA)

PADAA enhances responses to fine-grained and spatially varying anomalies.

Main characteristics:

- Dynamically enhances local anomaly responses
- Adapts to irregular anomaly structures
- Improves subtle defect localization
- Strengthens spatial sensitivity to heterogeneous anomalies

---

## Experimental Results

### MVTec AD

| Metric | Result |
|---|---|
| Image-level AUROC | 99.4 |
| Pixel-level AUROC | 98.2 |
| AUPRO | 94.9 |

CAUNet achieves state-of-the-art performance on MVTec AD and demonstrates strong localization capability for fine-grained industrial defects.

---

### Additional Benchmarks

The proposed framework also demonstrates strong generalization capability on:

- VisA
- MPDD

compared with existing industrial anomaly detection methods.

---

## Repository Structure

```text
CAUNet
│── data/                        # Dataset directory
│── datasets/                    # Dataset loading scripts
│── models/                      # Network architectures
│── modules/                     # Core modules of CAUNet
│── utils/                       # Utility functions
│── checkpoints/                 # Saved model weights
│── train.py                     # Training script
│── test.py                      # Evaluation script
│── requirements.txt             # Python dependencies
│── README.md                    # Project description
```

---

## Environment

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8

### Main Dependencies

```bash
torch
torchvision
numpy
opencv-python
scikit-learn
scipy
timm
Pillow
matplotlib
tqdm
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### MVTec AD

Download:

https://www.mvtec.com/company/research/datasets/mvtec-ad

Directory structure:

```text
data/
└── mvtec/
    ├── bottle/
    ├── cable/
    ├── capsule/
    └── ...
```

---

### VisA

Download:

https://github.com/amazon-science/spot-diff

---

### MPDD

Download:

https://github.com/stepanje/MPDD

---

## Training

Train CAUNet using:

```bash
python train.py \
    --dataset mvtec \
    --data_path ./data/mvtec \
    --batch_size 8 \
    --epochs 300
```

---

## Testing

Evaluate a trained model using:

```bash
python test.py \
    --dataset mvtec \
    --checkpoint ./checkpoints/model.pth
```

---

## Evaluation Metrics

The following metrics are used for evaluation:

- Image-level AUROC
- Pixel-level AUROC
- AUPRO
- F1-score

---

## Visualization

The framework supports visualization of:

- Input images
- Ground-truth masks
- Predicted anomaly maps
- Binary segmentation results

These visualizations help analyze localization quality and anomaly response behavior.

---

## Reproducibility

To reproduce the reported results:

1. Prepare the datasets correctly
2. Install all required dependencies
3. Train the model using the provided settings
4. Evaluate using released checkpoints

---

## Citation

If you find this repository useful for your research, please cite:

```bibtex
@article{CAUNet2026,
  title={Unified Anomaly Utilization Framework for Industrial Visual Defect Detection and Localization},
  author={Author Names},
  journal={The Visual Computer},
  year={2026}
}
```

---

## Acknowledgements

We sincerely thank the authors of the following works and datasets:

- PatchCore
- STPM
- DRAEM
- RD4AD
- MVTec AD
- VisA
- MPDD

---

## License

This repository is released for academic research purposes only.

---

## Contact

For questions, discussions, or collaborations, please open an issue on GitHub.

Project page:

https://github.com/xl-li-5913/CAUNet.git
