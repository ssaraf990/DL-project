# Data Directory

This directory manages raw ESC-50 audio data and processed feature artifacts.

## Dataset Setup Instructions

Place the official ESC-50 dataset files into `data/raw/` with the following structure:

```
data/
└── raw/
    ├── audio/              # Contains 2,000 original .wav audio files
    └── meta/
        └── esc50.csv       # Dataset metadata CSV file
```

## Dataset Specifications

- **Total Clips:** 2,000 audio files (5 seconds each).
- **Classes:** 50 classes (40 clips per class).
- **Metadata Columns:** `filename`, `fold`, `target`, `category`, `esc10`, `src_file`, `take`.

> **Git Policy Notice:** `.wav` audio files and preprocessed feature matrices (`.npy`, `.pt`) are ignored by `.gitignore` and must NOT be committed to the remote repository.
