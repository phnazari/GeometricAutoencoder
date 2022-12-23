import os

import geomstats.backend as gs
from torch.autograd.functional import jacobian
from functorch import jacfwd

from conf import device

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch

from util import batch_jacobian

from geomstats.geometry.riemannian_metric import RiemannianMetric


class PullbackMetric(RiemannianMetric):
    def __init__(self, dim, immersion, shape=None, signature=None, default_point_type=None):
        super().__init__(dim, shape=shape, signature=signature, default_point_type=default_point_type)
        self.immersion = immersion

    def normalize(self, v, point):
        """
        normalize vector at point
        :param v: vector
        :param point: point
        :return: normal vector
        """

        return v / self.norm(v, point)

    def metric_matrix_derivative(self, base_point=None):
        """
        Compute derivative of the inner prod matrix at base point.

        :param base_point : Base point
        :returns: matrix derivative
        """

        metric_derivative = batch_jacobian(self.metric_matrix, base_point)

        return metric_derivative

    def norm(self, vector, base_point=None, matrix=None):
        """
        Norm function that doesn't generate a new matrix because it already exists
        :param matrix: matrix for norm
        :param vector: vector which we want to calculate the norm of
        :return: norm
        """
        if matrix is None:
            return super().norm(vector, base_point)
        else:
            if vector.dim() > 1:
                result = torch.einsum("ijk,kl->ijl", matrix, vector)
                result = torch.einsum("mn,imn->in", vector, result)
                result = torch.sqrt(result)

                return result
            else:
                return torch.sqrt(vector @ matrix @ vector)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None, matrix=None):
        """
        Inner product if matrix is already calculated
        :param base_point: the basepoint under consideration
        :param matrix: metric
        :param tangent_vec_a: first vector
        :param tangent_vec_b: second vector
        :return: inner product
        """

        if matrix is None:
            return super().inner_product(tangent_vec_a, tangent_vec_b, base_point)
        else:
            prod = torch.einsum("...i,...ij,j->...", tangent_vec_a, matrix, tangent_vec_b)

            return prod

    def cometric_matrix(self, base_point=None, metric_matrix=None):
        """
        Inner co-product matrix at the cotangent space at a base point.

        :param base_point: the base point
        :param metric_matrix: the matrix to be inverted. If passed not computed twice
        :return : inverse of the metric matrix
        """

        if metric_matrix is None:
            metric_matrix = self.metric_matrix(base_point)

        # invert the batch of matrices
        inv_ex = torch.linalg.inv_ex(metric_matrix)
        cometric_matrix = inv_ex.inverse

        # remove the non-invertible matrices
        # projector = torch.ones_like(cometric_matrix)
        # problematics = inv_ex.info != 0
        # projector[problematics] = torch.zeros((2, 2), device=device)

        # auf nummer sicher: noch rausprojizieren
        # TODO: is the projector necessary?
        cometric_matrix = torch.nan_to_num(cometric_matrix, 0, 0, 0)  # * projector

        # if torch.any(problematics):
        #    print(f"{Color.RED}[WARNING] metric not invertible at points {base_point[problematics].data}{Color.NC}")

        return cometric_matrix

    def metric_matrix(self, base_point=None, **joblib_kwargs):
        """
        Calculates pullback of euclidian metric under f at point
        :param base_point: point at which we want to calculate the metric
        :return: pullback of euclidian metric under f at point
        """

        #print(base_point.shape)
        #print(self.immersion(base_point).shape)

        J = batch_jacobian(self.immersion, base_point)

        # TODO: remove this
        base_point = torch.squeeze(base_point)

        if base_point.dim() == 1:
            # TODO: remove squeezing?!
            J = torch.squeeze(J)
            metric = J.T @ J
        else:
            # TODO: does this squeezing have to be removed?!
            J = torch.squeeze(J)
            metric = torch.matmul(torch.transpose(J, 1, 2), J)

        return metric

    def christoffels(self, base_point):
        """
        Compute Christoffel symbols of the Levi-Civita connection.

        :param base_point : the base point
        :param cometric_matrix: the cometric matrix, so it doesn't have to be calculated multiple times
        :param metric_matrix_derivative: derivative of cometric matrix so just computed once
        :returns : Christoffels
        """

        cometric_matrix = self.cometric_matrix(base_point)
        metric_matrix_derivative = self.metric_matrix_derivative(base_point)

        term_1 = gs.einsum(
            # "...lk,...jli->...kij", cometric_matrix, metric_matrix_derivative
            "...kl,...jli->...kij", cometric_matrix, metric_matrix_derivative
        )
        term_2 = gs.einsum(
            # "...lk,...ilj->...kij", cometric_matrix, metric_matrix_derivative  # lij? no!
            "...kl,...ilj->...kij", cometric_matrix, metric_matrix_derivative
        )
        term_3 = -gs.einsum(
            # "...lk,...ijl->...kij", cometric_matrix, metric_matrix_derivative
            "...kl,...ijl->...kij", cometric_matrix, metric_matrix_derivative
        )

        christoffels = 0.5 * (term_1 + term_2 + term_3)

        return christoffels
