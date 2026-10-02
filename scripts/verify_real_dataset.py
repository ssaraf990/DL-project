"""
Real ESC-50 Dataset Verification Script.

Programmatically inspects and verifies:
1. File counts and metadata integrity of esc50.csv.
2. Target class counts, recordings per class, and fold distributions.
3. Audio properties (sample rate, channels, duration, readability) of real .wav files.
4. Dataset loader functionality across all 4 model feature representations.
5. Strict fold separation and zero data leakage.
"""

import os
import pandas as pd
import numpy as np
import soundfile as sf

from preprocessing.dataset_loader import ESC50Dataset


def verify_dataset_structure():
    meta_path = "data/raw/meta/esc50.csv"
    audio_dir = "data/raw/audio"

    print("=== 1. DATASET LOCATION & METADATA VERIFICATION ===")
    assert os.path.exists(meta_path), f"esc50.csv not found at {meta_path}"
    assert os.path.exists(audio_dir), f"audio dir not found at {audio_dir}"

    df = pd.read_csv(meta_path)
    print(f"✓ Metadata file loaded successfully: {len(df)} records.")

    expected_cols = {"filename", "fold", "target", "category", "esc10", "src_file", "take"}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"
    print(f"✓ Metadata columns verified: {list(df.columns)}")

    # Class & Fold distributions
    n_classes = df["target"].nunique()
    classes = df["category"].nunique()
    print(f"✓ Total unique classes: {n_classes} targets / {classes} categories.")
    assert n_classes == 50, f"Expected 50 classes, found {n_classes}"

    counts_per_class = df["category"].value_counts()
    min_count, max_count = counts_per_class.min(), counts_per_class.max()
    print(f"✓ Recordings per class: exactly {min_count} per class across all 50 classes.")
    assert min_count == 40 and max_count == 40, f"Expected 40 recordings per class, got min={min_count}, max={max_count}"

    fold_counts = df["fold"].value_counts().to_dict()
    print(f"✓ Fold distribution: {fold_counts}")
    assert set(fold_counts.keys()) == {1, 2, 3, 4, 5}, f"Expected folds 1..5, got {set(fold_counts.keys())}"
    assert all(c == 400 for c in fold_counts.values()), f"Expected 400 clips per fold, got {fold_counts}"

    return df, audio_dir, meta_path


def verify_audio_files(df, audio_dir):
    print("\n=== 2. AUDIO FILE INTEGRITY VERIFICATION ===")
    audio_files = os.listdir(audio_dir)
    wav_files = [f for f in audio_files if f.endswith(".wav")]
    print(f"✓ Total .wav files in audio directory: {len(wav_files)}")
    assert len(wav_files) == 2000, f"Expected 2000 wav files, found {len(wav_files)}"

    # Check file correspondence with metadata CSV
    csv_filenames = set(df["filename"])
    actual_filenames = set(wav_files)
    missing_files = csv_filenames - actual_filenames
    assert len(missing_files) == 0, f"Missing {len(missing_files)} files referenced in CSV!"
    print("✓ 100% filename match between CSV metadata and audio directory.")

    # Lightweight audio header inspection on a sample subset
    sample_files = wav_files[:50]
    sample_rates = set()
    channels_set = set()
    durations = []

    corrupt_count = 0
    for fname in wav_files:
        fpath = os.path.join(audio_dir, fname)
        try:
            info = sf.info(fpath)
            if fname in sample_files:
                sample_rates.add(info.samplerate)
                channels_set.add(info.channels)
                durations.append(info.duration)
        except Exception as e:
            corrupt_count += 1
            print(f"Corrupt file detected: {fname}, error: {e}")

    assert corrupt_count == 0, f"Found {corrupt_count} corrupt audio files!"
    print(f"✓ Corrupted audio files count: 0")
    print(f"✓ Native Sample Rate: {sample_rates} Hz (Official native rate is 44,100 Hz)")
    print(f"✓ Channels: {channels_set} (1 = Mono)")
    avg_duration = np.mean(durations)
    print(f"✓ Average Duration: {avg_duration:.2f} seconds")


def verify_dataset_loader_and_real_features(meta_path, audio_dir):
    print("\n=== 3. DATASET LOADER & REAL FEATURE SHAPE VERIFICATION ===")

    # Test Fold 5 as held-out validation set
    val_fold = 5
    train_ds_mel = ESC50Dataset(meta_path, audio_dir, val_fold=val_fold, is_train=True, feature_type="mel_spectrogram")
    test_ds_mel = ESC50Dataset(meta_path, audio_dir, val_fold=val_fold, is_train=False, feature_type="mel_spectrogram")

    print(f"✓ Training set size (val_fold={val_fold}): {len(train_ds_mel)} samples (expected 1600)")
    print(f"✓ Test/Validation set size (val_fold={val_fold}): {len(test_ds_mel)} samples (expected 400)")

    assert len(train_ds_mel) == 1600, f"Expected 1600 train samples, got {len(train_ds_mel)}"
    assert len(test_ds_mel) == 400, f"Expected 400 test samples, got {len(test_ds_mel)}"

    # Zero overlap check
    train_files = set(train_ds_mel.df["filename"])
    test_files = set(test_ds_mel.df["filename"])
    overlap = train_files.intersection(test_files)
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} files present in both sets."
    print("✓ Zero file overlap between training and test sets verified.")

    train_folds = set(train_ds_mel.df["fold"].unique())
    test_folds = set(test_ds_mel.df["fold"].unique())
    assert 5 not in train_folds, "Fold 5 leaked into training dataset!"
    assert test_folds == {5}, f"Test dataset contains unexpected folds: {test_folds}"
    print("✓ Strict fold isolation verified (Train folds: 1, 2, 3, 4 | Test fold: 5).")

    # Retrieve real audio features and verify shapes
    feature_configs = [
        ("mfcc", (40,)),
        ("mel_spectrogram", (1, 128, 216)),
        ("seq_mfcc", (216, 40)),
        ("hybrid", (1, 128, 216)),
    ]

    for ftype, expected_shape in feature_configs:
        ds = ESC50Dataset(meta_path, audio_dir, val_fold=5, is_train=False, feature_type=ftype)
        real_tensor, real_label = ds[0]
        assert real_tensor.shape == expected_shape, f"Feature '{ftype}' shape error: expected {expected_shape}, got {real_tensor.shape}"
        assert 0 <= real_label <= 49, f"Invalid label index: {real_label}"
        print(f"✓ Real Audio Feature '{ftype}' output shape: {real_tensor.shape} | Target Label: {real_label}")


if __name__ == "__main__":
    print("Starting Programmatic Verification of Real ESC-50 Dataset...\n")
    df, audio_dir, meta_path = verify_dataset_structure()
    verify_audio_files(df, audio_dir)
    verify_dataset_loader_and_real_features(meta_path, audio_dir)
    print("\n✓ ALL REAL DATASET VERIFICATION CHECKS PASSED SUCCESSFULLY!")
