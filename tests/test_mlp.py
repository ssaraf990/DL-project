"""Unit tests for the ESC-50 MLP architecture."""

import pytest
import torch

from models.mlp.model import ESC50MLP


def test_mlp_returns_raw_logits_for_50_classes() -> None:
    model = ESC50MLP()
    model.eval()
    logits = model(torch.randn(8, 40))
    assert logits.shape == (8, 50)
    assert torch.isfinite(logits).all()


def test_mlp_rejects_wrong_feature_shape() -> None:
    model = ESC50MLP()
    with pytest.raises(ValueError, match="expects input shape"):
        model(torch.randn(8, 39))
