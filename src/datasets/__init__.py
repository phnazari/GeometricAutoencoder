"""Datasets."""
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .cifar10 import CIFAR
from .pbmc import PBMC
from .celegans import CElegans
from .zilionis import Zilionis
from .earth import Earth

__all__ = ['MNIST', 'FashionMNIST', 'CIFAR', 'PBMC', 'CElegans', 'Zilionis', 'Earth']
