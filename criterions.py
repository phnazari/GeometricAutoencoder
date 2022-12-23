import torch

from conf import LOWER_EPSILON, UPPER_EPSILON, device

from diffgeo.manifolds import RiemannianManifold
from diffgeo.metrics import PullbackMetric
from diffgeo.connections import LeviCivitaConnection
from util import sample_points, distances, batch_jacobian
import torch

from conf import LOWER_EPSILON, UPPER_EPSILON, device

from diffgeo.manifolds import RiemannianManifold
from diffgeo.metrics import PullbackMetric
from diffgeo.connections import LeviCivitaConnection
from util import sample_points, distances, batch_jacobian


class Loss:
    def __init__(self, model=None):
        # a manifold object
        self.model = model
        pbm = PullbackMetric(2, model.decoder)
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
        if epoch >= 0:
            loss_det = self.determinant_loss()
        else:
            loss_det = torch.tensor([0.], device=device, requires_grad=True)

        return loss_det

    def determinant_loss(self):
        """
        MSE loss of log10 of determinants from log10 of the mean determinant
        :return: loss
        """

        dets = self.manifold.metric_det(base_point=self.model.latent_activations)

        # noinspection PyTypeChecker
        log_dets = torch.where((dets > LOWER_EPSILON) & (dets < UPPER_EPSILON),
                               torch.log10(dets),
                               torch.ones_like(dets))
        # dets = torch.maximum(dets, LOWER_EPSILON * torch.ones_like(dets))

        raw_loss = torch.var(log_dets)

        return raw_loss


class IndicatrixLoss(Loss):
    def __call__(self, epoch=0, *args, **kwargs):
        if epoch >= 0:
            loss_indicatrix = self.indicatrix_loss()
        else:
            loss_indicatrix = torch.tensor([0.], device=device)

        return loss_indicatrix

    def indicatrix_loss(self):
        # symmetric matrices have real eigenvalues
        e = torch.linalg.eigvals(self.manifold.metric.metric_matrix(self.model.latent_activations)).real

        loss = torch.sum((e - 1) ** 2) / e.numel()
        # loss = torch.var(e)

        return loss

        # loss = torch.sum((e - torch.mean(e)).pow(2))

        # compute relative deviation of smaller eigenvalue
        e_max = torch.max(e, dim=1).values
        e_diff = torch.abs(e[:, 0] - e[:, 1])

        relative_variance = torch.where(e_max != 0, e_diff / e_max, torch.zeros_like(e_diff))

        loss = torch.mean(relative_variance)

        return loss


class ChristoffelLoss(Loss):
    def __call__(self, epoch=0, *args, **kwargs):
        if epoch >= 0:
            loss_gamma_deriv = self.gamma_derivative_loss()
        else:
            loss_gamma_deriv = torch.tensor([0.], device=device, requires_grad=True)

        return loss_gamma_deriv

    def gamma_derivative_loss(self):
        return torch.max(self.manifold.christoffel_derivative(self.model.latent_activations))


class CurvatureLoss(Loss):
    def __init__(self, n_gaussian_samples=0., n_origin_samples=0., std=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gaussian_samples = n_gaussian_samples
        self.n_origin_samples = n_origin_samples
        self.std = std

    def __call__(self, epoch=0, *args, **kwargs):
        if epoch >= 0:
            loss_curv = self.curvature_loss()
        else:
            loss_curv = torch.tensor([0.], device=device, requires_grad=True)

        return loss_curv

    def curvature_loss(self, max_mag=None):
        """
        Loss resulting from the fact that sectional curvature varies in order of magnitude
        :return: loss
        """

        curv = self.manifold.sectional_curvature(torch.tensor([1., 0.], device=device),
                                                 torch.tensor([0., 1.], device=device),
                                                 base_point=self.model.latent_activations_detached)

        # samples_gaussian, samples_origin = self.sample_from_gaussian()
        # samples_gaussian = samples_gaussian.view(-1, 2)

        # calculate curvature
        # curv = self.manifold.sectional_curvature(torch.tensor([1., 0.], device=device),
        #                                         torch.tensor([0., 1.], device=device),
        #                                         base_point=samples_gaussian)
        # curv = curv.view(self.n_gaussian_samples, samples_origin.shape[0])

        asinh_curv = torch.asinh(curv)

        # calculate variance along first dimension and computing the mean
        raw_loss = torch.mean(torch.abs(asinh_curv), dim=0).mean()

        # raw_loss = torch.var(asinh_curv) * torch.var(self.model.latent_activations)

        if max_mag is not None:
            raw_loss = torch.minimum(raw_loss, 10 ** (max_mag - 1) * torch.ones_like(raw_loss))

        return raw_loss

    def sample_from_gaussian(self):
        if self.n_origin_samples:
            mean = sample_points(rep1=self.model.latent_activations_detached, num_samples=self.n_origin_samples)
        else:
            mean = self.model.latent_activations

        # scale standard deviation by determinant
        dets = self.manifold.metric_det(base_point=mean).detach_()
        dets_scaled = torch.where(dets > 0, torch.log10(dets), torch.ones_like(dets))

        # standard deviation should be positive
        dets_scaled = dets_scaled - torch.min(dets_scaled) + 1
        # dets_scaled[dets_scaled <= 0] = 1

        # scale standard deviation
        std = self.std / dets_scaled
        std = std.unsqueeze(0).unsqueeze(2).expand(self.n_gaussian_samples, *mean.shape)

        # expand the mean
        mean_expanded = mean.expand(self.n_gaussian_samples, *mean.shape)

        # draw samples
        samples = torch.normal(mean=mean_expanded, std=std)

        return samples, mean


class DistanceLoss(Loss):
    def __init__(self, n_dist_samples=0, *args, **kwargs):
        self.n_dist_samples = n_dist_samples
        super().__init__(*args, **kwargs)

    def __call__(self, outputs, epoch=0, *args, **kwargs):
        if epoch >= 0:
            loss_dist = self.distance_loss(outputs)
        else:
            loss_dist = torch.tensor([0.], device=device, requires_grad=True)

        return loss_dist

    def distance_loss(self, outputs):
        """
        Compute distances in latent space and output space for a sample and compute MSE error
        :param outputs: the outputs of the batch
        :return: loss
        """

        _, _, _, _, dist_pimg_norm, dist_img_norm = distances(rep1=self.model.latent_activations,
                                                              rep2=outputs,
                                                              num_samples=self.n_dist_samples)

        # TODO: kann ich hier nicht irgendwie mean(dist(pow())) machen?
        # TODO: den loss aus den latex notizen implementieren?!
        # TODO: wird der loss hier durch de- und encoder gepushed? Bzw. sollte er es werden?
        raw_loss = torch.sum((dist_img_norm - dist_pimg_norm) ** 2) / torch.numel(dist_img_norm)

        return raw_loss
