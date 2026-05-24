# CAUNet: Complete Anomaly Utilization Network for Industrial Anomaly Detection

Official implementation of **CAUNet** for industrial anomaly detection.

> **Paper**: *CAUNet: Complete Anomaly Utilization Network for Industrial Anomaly Detection*  
> **Journal**: *The Visual Computer*

---

## Overview

Industrial anomaly detection aims to identify defective regions and abnormal samples under a one-class learning setting where only normal training samples are available. Existing pseudo-anomaly-based methods usually rely on manually designed perturbations to construct anomaly supervision. However, these methods often suffer from three limitations:

1. Pseudo anomalies lack structural awareness.
2. Anomaly knowledge is difficult to transfer effectively into the normal feature space.
3. Feature responses are insufficiently adaptive to spatially varying defects.

To address these challenges, we propose **CAUNet (Complete Anomaly Utilization Network)**, a unified anomaly utilization framework that establishes a complete utilization chain for pseudo anomalies. The proposed framework improves anomaly generation, anomaly feature transfer, and adaptive anomaly perception simultaneously.

---

## Framework

### Main Components

CAUNet mainly contains the following modules:

- **Structure-aware pseudo anomaly generation**  
  Generates pseudo defects with richer structural diversity and more realistic spatial distributions.

- **Feature-level anomaly transfer mechanism**  
  Transfers anomaly knowledge from pseudo anomaly space into the normal feature representation space.

- **Adaptive anomaly response module**  
  Enhances spatial sensitivity to heterogeneous anomalies and improves localization capability.

- **Multi-scale feature interaction**  
  Aggregates semantic and detailed information across different feature resolutions.

---

## Experimental Results

CAUNet achieves competitive performance on multiple industrial anomaly detection benchmarks, including:

- MVTec AD
- VisA
- MPDD

The proposed framework demonstrates strong generalization capability and superior localization performance compared with existing methods.

---

## Repository Structure

```text
CAUNet
│── data/                       # Dataset directory
│── datasets/                   # Dataset loading scripts
│── models/                     # Network architectures
│── modules/                    # Core modules of CAUNet
│── utils/                      # Utility functions
│── checkpoints/                # Saved model weights
│── train.py                    # Training script
│── test.py                     # Evaluation script
│── requirements.txt            # Python dependencies
│── README.md                   # Project description
```

---

## Dependencies

### Environment

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8

### Main Libraries

```bash
torch
torchvision
numpy
opencv-python
scikit-learn
scipy
timm
Pillow
tqdm
matplotlib
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### MVTec AD

Download the dataset from:

https://www.mvtec.com/company/research/datasets/mvtec-ad

Organize the dataset as follows:

```text
data/
└── mvtec/
    ├── bottle/
    ├── cable/
    ├── capsule/
    └── ...
```

### VisA

Download the dataset from:

https://github.com/amazon-science/spot-diff

### MPDD

Download the dataset from:

https://github.com/stepanje/MPDD

---

## Training

Run the following command to train CAUNet:

```bash
python train.py \
    --dataset mvtec \
    --data_path ./data/mvtec \
    --batch_size 8 \
    --epochs 300
```

---

## Testing

Evaluate the trained model using:

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

## Key Algorithmic Details

### 1. Pseudo Anomaly Construction

CAUNet generates structurally diverse pseudo anomalies to simulate realistic industrial defects. Compared with simple handcrafted perturbations, the proposed strategy introduces richer spatial structures and improves anomaly representation learning.

### 2. Feature-level Anomaly Utilization

Instead of only using pseudo anomalies as pixel-level supervision, CAUNet further transfers anomaly-aware information into feature representations, enabling more discriminative anomaly perception.

### 3. Adaptive Spatial Response

The adaptive anomaly response mechanism dynamically adjusts feature responses according to spatially varying anomaly distributions, improving localization accuracy for irregular defects.

---

## Visualization

You can visualize anomaly maps and localization results during testing.

Example outputs include:

- Input image
- Ground-truth mask
- Predicted anomaly map
- Binary segmentation result

---

## Reproducibility

To reproduce the reported results:

1. Prepare the datasets correctly.
2. Install all dependencies.
3. Use the provided training configuration.
4. Evaluate using the released checkpoints.

---

## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{CAUNet2026,
  title={CAUNet: Complete Anomaly Utilization Network for Industrial Anomaly Detection},
  author={Author Names},
  journal={The Visual Computer},
  year={2026}
}
```

---

## Acknowledgements

This project benefits from previous studies on industrial anomaly detection and pseudo anomaly learning.

We sincerely thank the authors of:

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

For questions or collaborations, please open an issue or contact the authors through GitHub.
