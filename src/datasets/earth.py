"""Datasets."""
import os

from data import custom


class Earth(custom.Earth):
    """MNIST dataset."""

    def __init__(self, train=True):
        """Earth dataset."""
        super().__init__(train=train,
                         filename=os.path.join(os.path.dirname(__file__), '..', '..', "data/raw/earth/landmass.pt"))

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data.

        Args:
            x: Batch of data

        Returns:
            Tensor with normalization inversed.

        """
        normalized = 0.5 * (normalized + 1)
        return normalized
