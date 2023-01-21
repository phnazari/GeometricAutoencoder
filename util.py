"""
Utility functions
"""

import _pickle
import math
import os
from math import ceil

import torch

from mpl_toolkits.mplot3d import art3d

import numpy as np
from functorch import jacrev, jacfwd, vmap

import matplotlib
from matplotlib import cm, pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import juggle_axes
from scipy.optimize import fsolve
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from scipy.spatial import Delaunay

from conf import LOWER_EPSILON
from torch.autograd.functional import jacobian


class Color:
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    GREEN = '\033[0;32m'
    NC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_sc_kwargs():
    sc_kwargs = {
        "marker": ".",
        "alpha": .5,
        "s": 20,
        "edgecolors": None,
        "linewidth": 0.
    }

    return sc_kwargs


def get_saving_kwargs():
    kwargs = {
        "format": "png",
        "pad_inches": 0,
        "dpi": 120
    }

    return kwargs


def get_coordinates(latent_activations, grid=None, num_steps=20, coords0=None, model_name=None, dataset_name=None):
    """
    Get indicatrix positions
    Args:
        latent_activations: the embedding considered
        grid: the type of grid we consider
        num_steps: the number of steps in the orizontal direction
        coords0: one fixed coordinate that should be part of thhe grid
        model_name: the name of the model which created the embedding
        dataset_name: the name of the dataset considered

    Returns:
        None
    """

    x_min = torch.min(latent_activations[:, 0]).item()
    x_max = torch.max(latent_activations[:, 0]).item()
    y_min = torch.min(latent_activations[:, 1]).item()
    y_max = torch.max(latent_activations[:, 1]).item()

    # factor to scale the indicatrices by
    if model_name is None:
        factor = 0.3
    elif model_name == "ParametricUMAP":
        factor = 0.95
    elif model_name == "Vanilla":
        if dataset_name == "PBMC":
            factor = 0.12078598
            # factor = 0.5
            # factor = 0.3
        elif dataset_name == "Zilionis":
            factor = 0.05
        else:
            factor = 0.3
    else:
        factor = 0.3

    if num_steps != None:
        num_steps_x = num_steps
        num_steps_y = ceil((y_max - y_min) / (x_max - x_min) * num_steps_x)

        step_size_x = (x_max - x_min) / (num_steps_x)
        step_size_y = (y_max - y_min) / (num_steps_y)

    if grid == "dataset":
        coordinates = latent_activations
    elif grid == "off_data":
        coordinates = []
        xs = torch.linspace(x_min, x_max, steps=num_steps_x)
        ys = torch.linspace(y_min, y_max, steps=num_steps_y)
        num_xs = len(xs)
        num_ys = len(ys)

        num_tiles = len(xs) * len(ys)
        mean_data_per_tile = len(latent_activations) / num_tiles

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                mask_x = torch.logical_and(latent_activations[:, 0] >= x - step_size_x / 2,
                                           latent_activations[:, 0] <= x + step_size_x / 2)
                mask_y = torch.logical_and(latent_activations[:, 1] >= y - step_size_y / 2,
                                           latent_activations[:, 1] <= y + step_size_y / 2)
                mask = torch.logical_and(mask_x, mask_y)
                in_tile = latent_activations[mask].shape[0]

                max_data_per_tile = factor * mean_data_per_tile
                if (i == 0 or i == num_xs - 1) or (j == 0 or j == num_ys - 1):
                    if (i == 0 or i == num_xs - 1) and (j == 0 or j == num_ys - 1):
                        max_data_per_tile = max_data_per_tile / 4
                    else:
                        max_data_per_tile = max_data_per_tile / 2

                if in_tile < max_data_per_tile:
                    coordinates.append(torch.tensor([x, y]))

        coordinates = torch.stack(coordinates)
    elif grid == "on_data":
        if coords0 is not None:
            x_0 = coords0[0].item()
            y_0 = coords0[1].item()

            num_steps_left = int((x_0 - x_min) / (x_max - x_min) * num_steps_x)
            num_steps_right = num_steps - num_steps_left

            num_steps_up = int((y_max - y_0) / (y_max - y_min) * num_steps_y)
            num_steps_down = num_steps_y - num_steps_up

            x_left = x_0 - np.arange(num_steps_left) * step_size_x
            x_right = x_0 + np.arange(num_steps_right) * step_size_x

            y_down = y_0 - np.arange(num_steps_down) * step_size_y
            y_up = y_0 + np.arange(num_steps_up) * step_size_y

            x_left = np.flip(np.array(x_left))[:-1]
            x_right = np.array(x_right)
            y_up = np.array(y_up)
            y_down = np.flip(np.array(y_down))[:-1]

            xs = torch.from_numpy(np.concatenate((x_left, x_right))).float()
            ys = torch.from_numpy(np.concatenate((y_down, y_up))).float()

        else:
            xs = torch.linspace(x_min, x_max, steps=num_steps_x)
            ys = torch.linspace(y_min, y_max, steps=num_steps_y)

        num_tiles = len(xs) * len(ys)
        mean_data_per_tile = len(latent_activations) / num_tiles

        coordinates = []
        num_xs = len(xs)
        num_ys = len(ys)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                mask_x = torch.logical_and(latent_activations[:, 0] >= x - step_size_x / 2,
                                           latent_activations[:, 0] <= x + step_size_x / 2)
                mask_y = torch.logical_and(latent_activations[:, 1] >= y - step_size_y / 2,
                                           latent_activations[:, 1] <= y + step_size_y / 2)

                mask = torch.logical_and(mask_x, mask_y)
                in_tile = latent_activations[mask].shape[0]

                required_data_per_tile = factor * mean_data_per_tile
                if (i == 0 or i == num_xs - 1) or (j == 0 or j == num_ys - 1):
                    if (i == 0 or i == num_xs - 1) and (j == 0 or j == num_ys - 1):
                        required_data_per_tile = required_data_per_tile / 4
                    else:
                        required_data_per_tile = required_data_per_tile / 2

                if in_tile >= required_data_per_tile:
                    coordinates.append(torch.tensor([x, y]))

        coordinates = torch.stack(coordinates)
    elif grid == "convex_hull":
        coordinates = []
        xs = torch.linspace(x_min, x_max, steps=num_steps_x)
        ys = torch.linspace(y_min, y_max, steps=num_steps_y)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                coordinates.append(torch.tensor([x, y]))

        coordinates = torch.stack(coordinates)
    else:
        coordinates = None

    hull = get_hull(latent_activations)
    coordinates = coordinates[in_hull(coordinates, hull)]

    return coordinates


def get_hull(points):
    """
    Calculates the Delaunay hull for points
    :param points:
    :return:
    """

    return Delaunay(points)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    return hull.find_simplex(p) >= 0


def batch_jacobian(f, input):
    """
    Compute the diagonal entries of the jacobian of f with respect to x
    :param f: the function
    :param x: where it is to be evaluated
    :return: diagonal of df/dx. First dimension is the derivative
    """

    # compute vectorized jacobian. For curvature because of nested derivatives, for some of the backward functions
    # the forward mode AD is not implemented
    if input.ndim == 1:
        try:
            jac = jacfwd(f)(input)
        except NotImplementedError:
            jac = jacrev(f)(input)

    else:
        try:
            jac = vmap(jacfwd(f), in_dims=(0,))(input)
        except NotImplementedError:
            jac = vmap(jacrev(f), in_dims=(0,))(input)

    return jac


def symlog(x):
    """
    logarithm extended to negative reals
    """

    # sl = torch.where(x >= 1., torch.log10(x), torch.where(x <= -1., - torch.log10(-x), torch.zeros_like(x)))
    sl = torch.sign(x) * torch.log10(torch.abs(x) + 1)

    return sl


def symlog_inv(x):
    res = torch.where(x > 0, torch.pow(10, x), -torch.pow(10, x))

    return res


def minmax(item):
    return (item) / (torch.max(item) - torch.min(item)) + (torch.max(item) - torch.min(item)) / 2


def cmap_labels(labels, cmap=cm.turbo):
    """
    convert labels
    """
    # apply cmap and change base
    new_labels = (cmap(labels) * 255).astype(int)
    # remove opacity channel from rgba
    new_labels = torch.tensor(new_labels[:, :-1])

    return new_labels


def values_in_quantile(x, q=0):
    """
    Get alues in q quantile
    """
    if q == 1.:
        idx = torch.arange(len(x))
    else:
        largest_abs = torch.topk(torch.abs(x), k=int(q * len(x)), largest=True)
        smallest = torch.topk(largest_abs.values, k=int(len(largest_abs.values) / len(x) * q * len(largest_abs.values)),
                              largest=False)

        idx = largest_abs.indices[smallest.indices]

    return idx


def determine_scaling_fn(scaling):
    # determine scaling of curvature values
    scaling_fn = None
    if type(scaling) == str:
        if scaling == "asinh":
            scaling_fn = torch.asinh
        elif scaling == "lin":
            scaling_fn = lambda x: x
        elif scaling == "symlog":
            scaling_fn = symlog
        elif scaling == "log":
            scaling_fn = torch.log10
        else:
            print("TROW CUSTOM ERROR")
    elif callable(scaling):
        scaling_fn = scaling
    else:
        print("THROW CUSTOM ERROR")

    def inverse(x):
        if scaling == "asinh":
            return torch.sinh(x)
        elif scaling == "lin":
            return x
        elif scaling == "symlog":
            return symlog_inv(x)
        elif scaling == "log":
            return torch.pow(10, x)

        return x

    if scaling == "lin":
        prefix = ""
    else:
        prefix = f"{scaling} of "

    return scaling_fn, prefix


def transform_axes(ax, invisible=True):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    if invisible:
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_axis_off()

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)


def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[0, d[2], -d[1]],
                     [-d[2], 0, d[0]],
                     [d[1], -d[0], 0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle ** 2) * (eye - ddt) + sin_angle * skew
    return M


def pathpatch_2d_to_3d(pathpatch, z=0, normal='z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0, 0, 0), index)

    normal /= np.linalg.norm(normal)  # Make sure the vector is normalised

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = rotation_matrix(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta


def get_significant(val):
    i = 0
    in_sig = False
    while i < len(str(val)):
        if str(val)[i] not in [".", "0"]:
            in_sig = True

        if in_sig:
            if int(str(val)[i]) <= 2:
                return i + 1
            else:
                return i

        i += 1


def get_next_digit(val, i):
    val_10 = float(val) * 10 ** (i)
    return int(str(val_10).split(".")[1][0])


def round_significant(data, errors):
    """
    Round to first significant digit
    """
    results = []

    i = 0
    while i < len(data):
        dist = get_significant(errors[i])

        if errors[i] != 0.:
            dist = int(math.floor(math.log10(abs(errors[i]))))
        else:
            dist = 1

        if get_next_digit(errors[i], -dist) <= 2.:
            dist -= 1

        if errors[i] != 0:
            err = round(errors[i], -dist)
            val = round(data[i], -dist)
        else:
            err = round(errors[i], 1)
            val = round(data[i], 2)

        if err.is_integer():
            err = int(err)

        if val.is_integer():
            val = int(val)

        results.append(f'{val}' + ' $\pm$ ' + f'{err}')

        i += 1

    return results
