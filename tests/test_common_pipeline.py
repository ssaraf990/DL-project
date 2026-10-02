"""
Validation Test Suite for Common Preprocessing and Evaluation Pipeline.

Verifies signal shapes, feature extraction dimensions, fold filtering,
metrics computation, and confusion matrix plotting using synthetic waveforms and metadata.
NOTE: Uses synthetic test data only to validate pipeline correctness. Does NOT train models or evaluate trained weights.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import torch
import soundfile as sf

from preprocessing.reproducibility import set_seed
from preprocessing.audio_utils import load_audio, pad_or_truncate_audio, normalize_waveform
from preprocessing.feature_extraction import (
    extract_mfcc,
    extract_sequential_mfcc,
    extract_mel_spectrogram,
    extract_hybrid_spectrogram,
)
from preprocessing.dataset_loader import ESC50Dataset
from evaluation.metrics import compute_metrics
from evaluation.confusion_matrix import plot_confusion_matrix


def test_reproducibility():
    set_seed(42)
    val1 = np.random.randn(5)
    set_seed(42)
    val2 = np.random.randn(5)
    assert np.allclose(val1, val2), "Random seed setting failed reproducibility check."
    print("✓ [PASS] Reproducibility set_seed test passed.")


def test_audio_utilities():
    target_sr = 22050
    target_length = 110250  # 5 seconds at 22050 Hz

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "test_clip.wav")
        # Create a 4-second synthetic sine wave at 44100 Hz
        sample_rate_orig = 44100
        t = np.linspace(0, 4.0, int(sample_rate_orig * 4.0), endpoint=False)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(audio_path, sine_wave, sample_rate_orig)

        # Test load_audio
        waveform, sr = load_audio(audio_path, target_sr=target_sr)
        assert waveform.ndim == 1, f"Expected 1D waveform, got {waveform.ndim}D"
        assert sr == target_sr, f"Expected sampling rate {target_sr}, got {sr}"
        print(f"✓ [PASS] load_audio returned 1D waveform with sr={sr}.")

        # Test pad_or_truncate_audio
        padded_waveform = pad_or_truncate_audio(waveform, target_length=target_length)
        assert len(padded_waveform) == target_length, (
            f"Expected padded length {target_length}, got {len(padded_waveform)}"
        )
        print(f"✓ [PASS] pad_or_truncate_audio produced exactly {len(padded_waveform)} samples.")

        # Test normalize_waveform
        norm_waveform = normalize_waveform(padded_waveform)
        assert np.isclose(np.mean(norm_waveform), 0.0, atol=1e-5), "Mean normalization check failed."
        assert np.isclose(np.std(norm_waveform), 1.0, atol=1e-3), "Std normalization check failed."
        print("✓ [PASS] normalize_waveform produced zero mean and unit variance.")


def test_feature_extractions():
    # 5-second synthetic waveform at 22050 Hz
    waveform = np.random.randn(110250)

    # 1. MFCC for MLP
    mfcc = extract_mfcc(waveform, sr=22050, n_mfcc=40)
    assert mfcc.shape == (40,), f"Expected MFCC shape (40,), got {mfcc.shape}"
    print(f"✓ [PASS] extract_mfcc shape verified: {mfcc.shape}")

    # 2. Sequential MFCC for LSTM
    seq_mfcc = extract_sequential_mfcc(waveform, sr=22050, n_mfcc=40, hop_length=512)
    assert seq_mfcc.ndim == 2 and seq_mfcc.shape[1] == 40, (
        f"Expected Sequential MFCC shape (T, 40), got {seq_mfcc.shape}"
    )
    assert seq_mfcc.shape[0] == 216, f"Expected T=216 frames, got T={seq_mfcc.shape[0]}"
    print(f"✓ [PASS] extract_sequential_mfcc shape verified: {seq_mfcc.shape}")

    # 3. Mel-spectrogram for CNN
    mel_spec = extract_mel_spectrogram(waveform, sr=22050, n_mels=128, hop_length=512)
    assert mel_spec.shape == (128, 216), f"Expected Mel-spectrogram shape (128, 216), got {mel_spec.shape}"
    print(f"✓ [PASS] extract_mel_spectrogram shape verified: {mel_spec.shape}")

    # 4. Hybrid Spectrogram for CNN-LSTM
    hybrid_spec = extract_hybrid_spectrogram(waveform, sr=22050, n_mels=128, hop_length=512)
    assert hybrid_spec.shape == (128, 216), f"Expected Hybrid shape (128, 216), got {hybrid_spec.shape}"
    print(f"✓ [PASS] extract_hybrid_spectrogram shape verified: {hybrid_spec.shape}")


def test_dataset_loader_and_folds():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = os.path.join(tmpdir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        meta_rows = []
        # Generate synthetic metadata for 20 files across 5 folds (4 per fold)
        for i in range(20):
            filename = f"1-1000-A-{i}.wav"
            fold = (i % 5) + 1
            target = i % 50
            category = f"class_{target}"
            src_file = 1000 + (i // 5)
            take = "A"
            meta_rows.append({
                "filename": filename,
                "fold": fold,
                "target": target,
                "category": category,
                "src_file": src_file,
                "take": take
            })
            # Create synthetic 1-second audio file
            sf.write(os.path.join(audio_dir, filename), np.random.randn(22050), 22050)

        csv_path = os.path.join(tmpdir, "esc50.csv")
        pd.DataFrame(meta_rows).to_csv(csv_path, index=False)

        # Test Fold Filtering (Held-out val_fold = 5)
        train_ds = ESC50Dataset(csv_path, audio_dir, val_fold=5, is_train=True, feature_type="mel_spectrogram")
        test_ds = ESC50Dataset(csv_path, audio_dir, val_fold=5, is_train=False, feature_type="mel_spectrogram")

        assert len(train_ds) == 16, f"Expected 16 training samples, got {len(train_ds)}"
        assert len(test_ds) == 4, f"Expected 4 test samples, got {len(test_ds)}"

        # Check fold separation / overlap
        train_folds = set(train_ds.df["fold"].unique())
        test_folds = set(test_ds.df["fold"].unique())
        assert 5 not in train_folds, "Held-out fold 5 accidentally appeared in training set!"
        assert test_folds == {5}, f"Expected test folds {{5}}, got {test_folds}"
        assert len(train_folds.intersection(test_folds)) == 0, "Train and test folds overlap!"
        print("✓ [PASS] ESC50Dataset official fold splitting & leakage check verified.")

        # Test item shape outputs for all 4 feature types
        for ftype, expected_shape in [
            ("mfcc", (40,)),
            ("mel_spectrogram", (1, 128, 216)),
            ("seq_mfcc", (216, 40)),
            ("hybrid", (1, 128, 216)),
        ]:
            ds = ESC50Dataset(csv_path, audio_dir, val_fold=5, is_train=True, feature_type=ftype)
            x_tensor, y_label = ds[0]
            assert isinstance(x_tensor, torch.Tensor), f"Expected PyTorch Tensor for {ftype}"
            assert x_tensor.shape == expected_shape, (
                f"Feature '{ftype}' expected tensor shape {expected_shape}, got {x_tensor.shape}"
            )
            assert isinstance(y_label, int), f"Expected int target, got {type(y_label)}"
            print(f"✓ [PASS] ESC50Dataset feature_type='{ftype}' output shape: {x_tensor.shape}, label: {y_label}")


def test_evaluation_metrics_and_cm():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic test labels
        y_true = np.random.randint(0, 50, size=100)
        y_pred = np.random.randint(0, 50, size=100)

        # Test metrics
        metrics = compute_metrics(y_true, y_pred)
        expected_keys = {
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
        }
        assert set(metrics.keys()) == expected_keys, f"Metrics dictionary keys missing. Got {metrics.keys()}"
        print(f"✓ [PASS] compute_metrics returned all required keys: accuracy={metrics['accuracy']:.4f}")

        # Test confusion matrix plot
        save_path = os.path.join(tmpdir, "cm.png")
        class_names = [f"class_{i}" for i in range(50)]
        cm = plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)
        assert cm.shape == (50, 50), f"Expected confusion matrix shape (50, 50), got {cm.shape}"
        assert os.path.exists(save_path), "Confusion matrix plot image file was not created!"
        print("✓ [PASS] plot_confusion_matrix generated 50x50 plot image.")


if __name__ == "__main__":
    print("Running ESC-50 Common Pipeline Validation Test Suite...\n")
    test_reproducibility()
    test_audio_utilities()
    test_feature_extractions()
    test_dataset_loader_and_folds()
    test_evaluation_metrics_and_cm()
    print("\n✓ ALL VALIDATION TESTS PASSED SUCCESSFULLY!")
