"""
THIS FILE WAS TAKEN FROM https://github.com/BorgwardtLab/topological-autoencoders
"""


"""Datasets."""
import os

import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets
from torchvision import transforms

BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw'))


class MNIST:
    """MNIST dataset."""

    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __init__(self, train=True):
        """MNIST dataset."""

        if train is True:
            self.dataset = ConcatDataset(
                [datasets.MNIST(BASEPATH, transform=self.transforms, train=True, download=True),
                 datasets.MNIST(BASEPATH, transform=self.transforms, train=False, download=True)])

        else:
            self.dataset = datasets.MNIST(
                BASEPATH, transform=self.transforms, train=False, download=True)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data.

        Args:
            x: Batch of data

        Returns:
            Tensor with normalization inversed.

        """
        normalized = 0.5 * (normalized + 1)
        return normalized
