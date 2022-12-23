import os
import traceback

import numpy as np
import geomstats.backend as gs

from conf import device, LOWER_EPSILON, BIGGER_LOWER_EPSILON, SMALLER_UPPER_EPSILON

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch
from functorch import jacrev

from util import preimage, Color, batch_jacobian

from geomstats.geometry.manifold import Manifold


class RiemannianManifold(Manifold):
    """
    Class for manifolds.

    :param dim : intd
            Dimension of the manifold.
    :param shape : tuple of int
            Shape of one element of the manifold.
            Optional, default : None.
    :param metric : RiemannianMetric
            Metric object to use on the manifold.
    :param default_point_type : str, {\'vector\', \'matrix\'}
            Point type.
            Optional, default: 'vector'.
    :param default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
            Coordinate type.
            Optional, default: 'intrinsic'.
    """

    def __init__(self, dim, shape, metric=None, connection=None, default_point_type=None,
                 default_coords_type="intrinsic", **kwargs):
        super().__init__(dim, shape, metric=metric, default_point_type=default_point_type,
                         default_coords_type=default_coords_type, **kwargs)

        self.connection = connection

    def christoffel_derivative(self, base_point=None):
        """
        Calculate the derivative of the christoffel symbols
        :return: derivative of christoffel
        """

        gamma_derivative = batch_jacobian(self.connection.christoffels, base_point)

        return gamma_derivative

    def metric_det(self, base_point=None):
        """
        Calculate the determinant of the metric matrix at base_point
        :param base_point: the point under consideration
        :param metric_matrix: the metric at point base_point
        :return: the determinant
        """

        metric_matrix = self.metric.metric_matrix(base_point=base_point)

        det = torch.linalg.det(metric_matrix)

        return det

    def riemannian_curvature_tensor(self, base_point=None):
        """
        Returns the curvature tensor symbols
        :param base_point: the base point
        :return: the coordinates of the curvature tensor, contravariant index in the first dimension
        """

        gamma = self.connection.christoffels(base_point)

        gamma_derivative = self.christoffel_derivative(base_point=base_point)

        term_1 = torch.einsum("...ljki->...lijk", gamma_derivative)
        term_2 = torch.einsum("...likj->...lijk", gamma_derivative)
        term_3 = torch.einsum("...mjk,...lim->...lijk", gamma, gamma)
        term_4 = torch.einsum("...mik,...ljm->...lijk", gamma, gamma)

        R = term_1 - term_2 + term_3 - term_4

        return R

    def riemannian_curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point=None):
        """
        :param tangent_vec_a:
        :param tangent_vec_b:
        :param tangent_vec_c:
        :param base_point:
        :return:
        """

        R = self.riemannian_curvature_tensor(base_point=base_point)

        s = torch.einsum("...lijk,i->...ljk", R, tangent_vec_a)
        s = torch.einsum("...ljk,j->...lk", s, tangent_vec_b)
        s = torch.einsum("...lk,k->...l", s, tangent_vec_c)

        return s

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Compute the sectional curvature
        :param tangent_vec_a: first vector
        :param tangent_vec_b: second vector
        :param base_point: the point under consideration
        :return: sectional curvature at base_point
        """

        if base_point.ndim == 1:
            base_point = torch.unsqueeze(base_point, 0)

        metric = self.metric.metric_matrix(base_point)
        # aab, aba, baa
        # bba, bab, abb
        curvature = self.riemannian_curvature(tangent_vec_a, tangent_vec_b, tangent_vec_b, base_point)

        sectional = self.metric.inner_product(curvature, tangent_vec_a, matrix=metric)

        # norm_a = self.metric.norm(tangent_vec_a, matrix=metric)
        # norm_b = self.metric.norm(tangent_vec_b, matrix=metric)
        norm_a = self.metric.inner_product(tangent_vec_a, tangent_vec_a, matrix=metric)
        norm_b = self.metric.inner_product(tangent_vec_b, tangent_vec_b, matrix=metric)
        inner_ab = self.metric.inner_product(tangent_vec_a, tangent_vec_b, matrix=metric)

        normalization_factor = norm_a * norm_b - inner_ab ** 2

        result = torch.where(normalization_factor != 0, sectional / normalization_factor, torch.zeros_like(sectional))

        return result

    def generate_unit_vectors(self, n, base_point):
        """
        calculate polygon lengths using metric
        :param n: number of vectors
        :param base_point: the base point
        :return: array of norms
        """

        # the angles
        phi = torch.linspace(0., 2 * np.pi, n, device=device)

        # generate circular vector patch
        raw_vectors = torch.stack([torch.sin(phi), torch.cos(phi)])

        # metric at the point
        metric = self.metric.metric_matrix(base_point)

        # normalize vectors in pullback metric
        norm_vectors = self.metric.norm(raw_vectors, matrix=metric)

        norm_vectors = norm_vectors.unsqueeze(2).expand(*norm_vectors.shape, raw_vectors.shape[0])

        # reshape the raw vectors
        raw_vectors = raw_vectors.unsqueeze(2).expand(*raw_vectors.shape, base_point.shape[0])
        raw_vectors = torch.transpose(raw_vectors, dim0=0, dim1=2)

        # normalize the vector patches
        unit_vectors = torch.where(norm_vectors != 0, raw_vectors / norm_vectors, torch.zeros_like(raw_vectors))

        return unit_vectors, norm_vectors

    def belongs(self, point, atol=gs.atol):
        return

    def is_tangent(self, vector, base_point, atol=gs.atol):
        return

    def random_point(self, n_samples=1, bound=1.0):
        return

    def to_tangent(self, vector, base_point):
        return
