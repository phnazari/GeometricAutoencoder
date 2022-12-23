"""Datasets."""
import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from AutoEncoderVisualization.data import custom


class Images(custom.Images):
    """MNIST dataset."""

    mean_channels = (0.131,)
    std_channels = (0.308,)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """MNIST dataset normalized."""

        if train:
            super().__init__(
                img1_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/images/image1.JPEG",
                img2_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/images/image2.JPEG",
                n_samples=80000,
                train=train)
        else:
            super().__init__(
                img1_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/images/image1.JPEG",
                img2_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/images/image2.JPEG",
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
