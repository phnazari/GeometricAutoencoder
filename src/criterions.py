"""
The Determinant Regularizer
"""

import torch

from conf import LOWER_EPSILON, UPPER_EPSILON, device

from src.diffgeo.manifolds import RiemannianManifold
from src.diffgeo.metrics import PullbackMetric
from src.diffgeo.connections import LeviCivitaConnection


class Loss:
    """
    A Basis class for custom loss functions
    """
    def __init__(self, model=None):
        # a manifold object
        self.model = model
        pbm = PullbackMetric(2, model.immersion)
        lcc = LeviCivitaConnection(2, pbm)
        self.manifold = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    @staticmethod
    def sample_points(latent_activations=None, outputs=None, num_samples=1):
        """
        Randomly sample points from batch in latent space and map it to the output space
        :param outputs: batch in output space
        :param latent_activations: batch in latent space
        :param num_samples: number of samples to take
        :return: (points in latent space, points in output space)
        """

        # randomly sample the points
        rand_choice = torch.randperm(latent_activations.shape[0])[:num_samples]

        pimg_origin = latent_activations[rand_choice, :]

        if outputs is None:
            return pimg_origin

        img_origin = outputs[rand_choice, :]

        return img_origin, pimg_origin


class DeterminantLoss(Loss):
    def __call__(self, epoch=0, *args, **kwargs):
        """
            Our Determinant Regularizer
        Args:
            epoch: current epoch
        Returns:

        """

        # here you can control whether the regularizer should only be switched on after a certain epoch
        if epoch >= 0:
            loss_det = self.determinant_loss()
        else:
            loss_det = torch.tensor([0.], device=device, requires_grad=True)

        return loss_det

    def determinant_loss(self):
        """
        Calculate the actual loss
        Returns:
            The determinant loss
        """

        # calculate the generalized jacobian determinant
        dets = self.manifold.metric_det(base_point=self.model.latent_activations)

        # noinspection PyTypeChecker
        log_dets = torch.where((dets > LOWER_EPSILON) & (dets < UPPER_EPSILON),
                               torch.log10(dets),
                               torch.ones_like(dets))

        # calculate the variance of the logarithm of the generalized jacobian determinant
        raw_loss = torch.var(log_dets)

        return raw_loss
