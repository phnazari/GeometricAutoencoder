"""
THIS FILE WAS PARTLY TAKEN FROM https://github.com/BorgwardtLab/topological-autoencoders
"""


"""Submodules used by models."""


# Hush the linter: Warning W0221 corresponds to a mismatch between parent class
# method signature and the child class
# pylint: disable=W0221




import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from collections import OrderedDict
from umap import ParametricUMAP
from .base import AutoencoderModel
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Print(nn.Module):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def forward(self, x):
        print(self.name, x.size())
        return x


class BoxAutoEncoder(AutoencoderModel):
    """100-100-100-2-100-100-100."""

    def __init__(self, input_dims=(1, 28, 28), **kwargs):
        super().__init__()
        self.latent_dim = 2
        n_input_dims = np.prod(input_dims)
        self.input_dim = n_input_dims.item()
        self.input_dims = input_dims

        self.encoder = nn.Sequential(
            # View((-1, n_input_dims)),
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.latent_dim, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=self.input_dim),
            # View((-1,) + tuple(input_dims)),
        )

        self.reconst_error = nn.MSELoss()

        self.register_hook()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        x = x.view((-1, self.input_dim))

        x = self.encoder(x)
        return x

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        z = self.decoder(z)
        z = z.view((-1, *self.input_dims))
        return z

    def forward_(self, x):
        x = x.view((-1, self.input_dim))
        x = self.encoder(x)

        z = self.decoder(x)
        z = z.view((-1, *self.input_dims))

        return z

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}

    def register_hook(self):
        self.encoder.register_forward_hook(self.get_activation())

    def get_activation(self):
        """
        :return: activations at layer model
        """

        def hook(model, input, output):
            self.latent_activations = output
            self.latent_activations_detached = output.detach()

        return hook

    def load(self, path):
        dict = torch.load(path)

        # edit keys, since they are created from higher TopoAE class
        new_dict = OrderedDict([])

        for key in dict.keys():
            arr = key.split(".")
            if len(arr) == 1 or arr[-1] not in ["weight", "bias"]:
                continue
                # del dict[key]
            # elif arr[-1] not in ["weight", "bias"]:
            #    del dict[key]
            else:
                if arr[0] == "autoencoder":
                    new_key = ".".join(arr[1:])
                else:
                    new_key = key

                new_dict.update({new_key: dict[key]})

                # dict = OrderedDict([(new_key, v) if k == key else (k, v) for k, v in d.items()])
                # new_dict.update({new_key: dict[key]})

        # self.load_state_dict(torch.load(path))
        self.load_state_dict(new_dict)

        self.eval()


class ConvolutionalAutoEncoder(AutoencoderModel):
    """100-100-100-2-100-100-100."""

    def __init__(self, input_dims=(1, 28, 28), **kwargs):
        super().__init__()
        self.latent_dim = 2
        n_input_dims = np.prod(input_dims)
        self.input_dim = n_input_dims.item()
        self.input_dims = input_dims

        self.encoder = nn.Sequential(
            # b, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ELU(),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.Conv2d(8, 2, 2, stride=1, padding=0),  # b, 2, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 8, 2, stride=1),  # b, 8, 2, 2
            nn.ELU(),
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ELU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

        self.reconst_error = nn.MSELoss()

        self.register_hook()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward_(self, x):
        x = self.encoder(x)
        z = self.decoder(x)

        return z

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}

    def register_hook(self):
        self.encoder.register_forward_hook(self.get_activation())

    def get_activation(self):
        """
        :return: activations at layer model
        """

        def hook(model, input, output):
            self.latent_activations = output
            self.latent_activations_detached = output.detach()

        return hook

    def load(self, path):
        dict = torch.load(path)

        # edit keys, since they are created from higher TopoAE class
        new_dict = OrderedDict([])

        for key in dict.keys():
            arr = key.split(".")
            if len(arr) == 1 or arr[-1] not in ["weight", "bias"]:
                continue
                # del dict[key]
            # elif arr[-1] not in ["weight", "bias"]:
            #    del dict[key]
            else:
                if arr[0] == "autoencoder":
                    new_key = ".".join(arr[1:])
                else:
                    new_key = key

                new_dict.update({new_key: dict[key]})

                # dict = OrderedDict([(new_key, v) if k == key else (k, v) for k, v in d.items()])
                # new_dict.update({new_key: dict[key]})

        # self.load_state_dict(torch.load(path))
        self.load_state_dict(new_dict)

        self.eval()


class LinearAE(AutoencoderModel):
    """input dim - 2 - input dim."""

    def __init__(self, input_dims=(1, 28, 28)):
        super().__init__()
        self.input_dims = input_dims
        n_input_dims = np.prod(input_dims)
        self.encoder = nn.Sequential(
            View((-1, n_input_dims)),
            nn.Linear(n_input_dims, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, n_input_dims),
            View((-1,) + tuple(input_dims)),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}
