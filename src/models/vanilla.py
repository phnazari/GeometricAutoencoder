"""Vanilla models."""
import torch.nn as nn
from criterions import DeterminantLoss

from src.models import submodules
from .base import AutoencoderModel


class ConvolutionalAutoencoderModel(submodules.ConvolutionalAutoencoder):
    """Convolutional autoencoder model.

    Same as the submodule but returns MSE loss.
    """

    def __init__(self):
        """Convolutional Autoencoder."""
        super().__init__()
        self.reconst_error = nn.MSELoss()

    def forward(self, x):
        """Return MSE reconstruction loss of convolutional autoencoder."""
        _, reconst = super().forward(x)
        return self.reconst_error(x, reconst), tuple()


class VanillaAutoencoderModel(AutoencoderModel):
    def __init__(self, autoencoder_model='ConvolutionalAutoencoder',
                 ae_kwargs=None):
        super().__init__()

        ae_kwargs = ae_kwargs if ae_kwargs else {}
        self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)

        self.with_geom_loss = False

        if self.with_geom_loss:
            self.determinant_criterion = DeterminantLoss(model=self.autoencoder)

    def forward(self, x):
        # Use reconstruction loss of autoencoder

        if self.with_geom_loss:
            ae_loss, ae_loss_comp = self.autoencoder(x)

            det_loss = self.determinant_criterion().detach_()

            loss = ae_loss
            loss_components = {
                'loss.autoencoder': ae_loss,
                'loss.geom_error': det_loss
            }

            loss_components.update(ae_loss_comp)
            return (
                loss,
                loss_components
            )
        else:
            return self.autoencoder(x)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)
