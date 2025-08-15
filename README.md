# BCGM: A Combined Approach for Solving Imbalance in Multimodal Emotion Recognition

## Abstract

This repository contains the implementation of **BCGM** (Balanced Classifier-Guided Modulation), a novel approach designed to address class imbalance challenges in multimodal emotion recognition tasks. Our method demonstrates improved performance on benchmark datasets through innovative combination strategies.

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Datasets](#datasets)
- [Results](#results)
- [Acknowledgement](#Acknowledgments)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/BCGM.git
cd BCGM
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Preprocessing Instructions

For detailed preprocessing steps, please refer to the original implementation from [OGM-GE_CVPR2022](https://github.com/GeWu-Lab/OGM-GE_CVPR2022).

### Download Preprocessed Datasets

We provide preprocessed versions of the datasets for convenience:

- **[CREMA-D](https://www.kaggle.com/datasets/biminhco/cremad)**: Multimodal emotion recognition dataset
- **[CMU-MOSI](https://www.kaggle.com/datasets/biminhco/dataset-mosi)**: Multimodal Opinion Sentiment and Emotion Intensity dataset

> **Important**: After downloading, update the dataset path in the configuration files to match your local directory structure.

## Usage

### Training

Navigate to the code directory before running training scripts:

```bash
cd code
```

#### CREMA-D Dataset
```bash
bash scripts/cremad/train_bcgm.sh
```

#### CMU-MOSI Dataset
```bash
bash scripts/mosi/train_bcgm.sh
```

### Testing

#### CREMA-D Dataset
```bash
bash scripts/cremad/inference.sh
```

#### CMU-MOSI Dataset
```bash
bash scripts/mosi/inference.sh
```

## Datasets

### CREMA-D
The **Crowdsourced Emotional Multimodal Actors Dataset** contains multimodal emotional expressions from actors, providing a comprehensive benchmark for emotion recognition research.

### CMU-MOSI
The **Carnegie Mellon University Multimodal Opinion Sentiment and Emotion Intensity** dataset offers rich multimodal data for sentiment analysis and emotion recognition tasks.

## Results

| Dataset | Accuracy | F1-Score | mAP   |
|---------|----------|----------|-------|
| CREMA-D | 87.91    | --       | 81.32 |
| CMU-MOSI| 78.40    | 78.30    | --    |

*Please update with your experimental results*

## Acknowledgments

- Original preprocessing methodology from [OGM-GE_CVPR2022](https://github.com/GeWu-Lab/OGM-GE_CVPR2022)
- Dataset providers: CREMA-D and CMU-MOSI research teams

---

For questions or issues, please open an issue in this repository or contact [21020240@vnu.edu.vn](mailto:21020240@vnu.edu.vn).
