"""Datasets."""
import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from data import custom


class PBMC(custom.PBMC):
    """MNIST dataset."""

    def __init__(self, train=True):
        """MNIST dataset normalized."""
        super().__init__(dir_path=os.path.join(os.path.dirname(__file__), '..', '..', "data/raw/pbmc"),
                         train=train)

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data.

        Args:
            x: Batch of data

        Returns:
            Tensor with normalization inversed.

        """
        normalized = 0.5 * (normalized + 1)
        return normalized
