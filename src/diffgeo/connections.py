import os

import geomstats.backend as gs

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch
from geomstats.geometry.connection import Connection


class LeviCivitaConnection(Connection):
    def __init__(self, dim, metric, shape=None, default_point_type=None, default_coords_type="intrinsic"):
        """
        :param metric: instance of PullbackMetric which gives the christoffel symbols
        """
        super().__init__(dim, shape=shape, default_point_type=default_point_type,
                         default_coords_type=default_coords_type)

        self.metric = metric

    def christoffels(self, base_point):
        return self.metric.christoffels(base_point)

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Generate parameterized function for the geodesic curve.

        :param initial_point : initial point of the geodesic.
        :param end_point : end point of the geodesic
        :param initial_tangent_vec :initial speed of the geodesics.

        :return
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents the different initial conditions, and the second
            corresponds to time.
        """
        point_type = self.default_point_type

        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            shooting_tangent_vec = self.log(point=end_point, base_point=initial_point)
            if initial_tangent_vec is not None:
                if not gs.allclose(shooting_tangent_vec, initial_tangent_vec):
                    raise RuntimeError(
                        "The shooting tangent vector is too"
                        " far from the input initial tangent vector."
                    )
            initial_tangent_vec = shooting_tangent_vec

        if point_type == "vector":
            initial_point = gs.to_ndarray(initial_point, to_ndim=2)
            initial_tangent_vec = gs.to_ndarray(initial_tangent_vec, to_ndim=2)

        else:
            initial_point = gs.to_ndarray(initial_point, to_ndim=3)
            initial_tangent_vec = gs.to_ndarray(initial_tangent_vec, to_ndim=3)
        n_initial_conditions = initial_tangent_vec.shape[0]

        if n_initial_conditions > 1 and len(initial_point) == 1:
            initial_point = gs.stack([initial_point[0]] * n_initial_conditions)

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_points,]
                Times at which to compute points of the geodesics.
            """

            t = gs.array(t)
            t = gs.cast(t, initial_tangent_vec.dtype)
            t = gs.to_ndarray(t, to_ndim=1)

            if point_type == "vector":
                tangent_vecs = gs.einsum("i,...k->...ik", t, initial_tangent_vec)
            else:
                tangent_vecs = gs.einsum("i,...kl->...ikl", t, initial_tangent_vec)

            initial_points = initial_point.repeat(tangent_vecs.shape[1], 1)

            points_at_time_t = [
                self.exp(tv, pt, step="rk4") for tv, pt in zip(tangent_vecs[0], initial_points)
            ]

            points_at_time_t = torch.vstack(points_at_time_t)

            return torch.squeeze(points_at_time_t)

        return path
