"""Datasets."""
import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from AutoEncoderVisualization.data import custom


class Zilionis(custom.Zilionis):
    """MNIST dataset."""

    def __init__(self, train=True):
        """MNIST dataset normalized."""
        super().__init__(dir_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/zilionis",
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
