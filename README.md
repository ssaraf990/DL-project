# Comparative Analysis of Deep Learning Architectures for Environmental Sound Classification

## Project Overview
Environmental Sound Classification (ESC) is a critical task in computational audio analysis with applications ranging from smart city monitoring and surveillance to context-aware personal devices. This project presents a systematic comparative evaluation of four distinct deep learning architectures—Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM—on the ESC-50 dataset.

The objective of this research is to analyze how varying feature representations (MFCCs, Mel-Spectrograms, Sequential MFCCs, and Time-Frequency representations) interact with specific neural network topologies to influence classification accuracy, model efficiency, and generalization capability.

> **Status Notice:** The project structure and workflow pipelines are currently initialized. Model implementation, training, and experimental evaluation will follow in subsequent phases.

---

## Dataset: ESC-50
The **ESC-50** dataset is a benchmark collection of 2,000 labeled environmental audio recordings.
- **Total Audio Clips:** 2,000 (5 seconds each)
- **Classes:** 50 distinct classes (40 clips per class)
- **Categories:** 5 major domains:
  1. Animals
  2. Natural soundscapes & water sounds
  3. Human, non-speech sounds
  4. Interior / domestic sounds
  5. Exterior / urban noises
- **Standard Protocol:** 5-fold cross-validation scheme predefined by the dataset authors.

---

## Proposed Model Architectures & Input Representations

| Model | Assigned Team Member | Input Feature Representation | Description |
| :--- | :--- | :--- | :--- |
| **MLP** | Mridul Arora | MFCC (Mel-Frequency Cepstral Coefficients) | Global summary of spectral envelope features extracted across fixed time windows. |
| **CNN** | Satyam Saraf | Mel-spectrogram | 2D time-frequency log-Mel spectrogram images to capture spatial pattern features. |
| **LSTM** | Ryan Gupta | Sequential MFCC | Temporal sequence of frame-level MFCC vectors to model time-series dependencies. |
| **CNN-LSTM** | Karan Shah | Spectrogram / Time-Frequency Hybrid | Hybrid feature representation passed into CNN layers for local feature extraction, followed by LSTM layers for temporal sequence modeling. |

---

## Evaluation Metrics
All models will be rigorously evaluated under identical 5-fold cross-validation setups using the following metrics:
- **Accuracy:** Overall proportion of correctly identified audio clips.
- **Precision (Macro & Weighted):** Measurement of class-specific positive predictive value.
- **Recall (Macro & Weighted):** Sensitivity across environmental sound categories.
- **F1-Score (Macro & Weighted):** Harmonic mean balancing precision and recall.
- **Confusion Matrix:** Detailed class-level error distribution analysis.

---

## Team Members & Responsibilities

| Name | Student ID | Model Architecture | Role |
| :--- | :--- | :--- | :--- |
| **Mridul Arora** | 230953330 | MLP | Feature engineering (MFCC), baseline MLP model architecture |
| **Satyam Saraf** | 230953178 | CNN | Mel-spectrogram generation, 2D CNN model design |
| **Ryan Gupta** | 230953288 | LSTM | Sequential MFCC extraction, recurrent network implementation |
| **Karan Shah** | 230953540 | CNN-LSTM | Hybrid spectrogram feature pipelines, CNN-LSTM model integration |

---

## Project Directory Structure

```
DL-project/
│
├── data/
│   ├── raw/                  # Original un-processed ESC-50 audio files
│   ├── processed/            # Extracted features (MFCC, Mel-spectrograms, etc.)
│   └── README.md
│
├── preprocessing/
│   ├── __init__.py
│   ├── audio_utils.py        # Core audio loading & signal manipulation
│   ├── feature_extraction.py # Generators for MFCCs, Mel-spectrograms, and sequence features
│   ├── dataset_loader.py     # ESC-50 dataset parser & PyTorch DataLoaders
│   └── README.md
│
├── models/
│   ├── mlp/                  # MLP model architecture & configs
│   ├── cnn/                  # CNN model architecture & configs
│   ├── lstm/                 # LSTM model architecture & configs
│   └── cnn_lstm/             # CNN-LSTM hybrid model architecture & configs
│
├── experiments/              # Training scripts, configuration files, & run logs
│   ├── mlp/
│   ├── cnn/
│   ├── lstm/
│   └── cnn_lstm/
│
├── results/                  # Training metrics, weights, evaluation plots
│   ├── mlp/
│   ├── cnn/
│   ├── lstm/
│   └── cnn_lstm/
│
├── evaluation/
│   ├── metrics.py            # Classification metrics & performance scoring
│   ├── confusion_matrix.py   # Plotting and confusion matrix utilities
│   └── README.md
│
├── notebooks/                # Exploratory analysis & visualization notebooks
│
├── docs/                     # Project documentation & final report assets
│   ├── methodology/
│   ├── literature_review/
│   └── figures/
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and guidelines
└── .gitignore                # Git exclusion rules
```

---

## Getting Started

### Prerequisites
Ensure Python 3.8+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/ssaraf990/DL-project.git
cd DL-project
pip install -r requirements.txt
```

---

## License & Usage
This repository is developed for educational and academic research purposes as part of the Deep Learning coursework.
