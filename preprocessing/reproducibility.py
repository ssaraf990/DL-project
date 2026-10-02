"""
Reproducibility Utilities for ESC-50 Environmental Sound Classification.

Provides set_seed functionality to enforce deterministic execution across Python,
NumPy, PyTorch, and CUDA platforms.
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and PyTorch.

    Args:
        seed (int): Seed number to set. Defaults to 42.

    Notes on Determinism Tradeoffs:
        - Setting torch.backends.cudnn.deterministic = True ensures reproducibility on GPUs,
          but may slightly decrease throughput for certain convolutional neural network operations.
        - setting PYTHONHASHSEED ensures dictionary and set ordering remains deterministic across runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
