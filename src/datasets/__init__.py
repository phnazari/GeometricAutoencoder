"""Datasets."""
from .manifolds import Spheres
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .cifar10 import CIFAR
from .pbmc import PBMC
from .celegans import CElegans
from .zilionis import Zilionis
from .images import Images
from .earth import Earth

__all__ = ['Spheres', 'MNIST', 'FashionMNIST', 'CIFAR', 'PBMC', 'CElegans', 'Zilionis', 'Images', 'Earth']
