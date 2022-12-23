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


def preimage(func, image, x0=np.array([0]), xtol=1e-2, t0=-1, maxfev=100):
    """
    return root of func
    :param func: function considered
    :param x0: initial guess for root
    :param image: point we want to find the preimage of
    :param xtol: tolerance of derivation inbetween two iterations
    :param t0: old time, we don't want to go backwards in time!
    :return:
    """

    # TODO: add good guess, point plus the vector along geodesic!
    # TODO: add fprime
    # TODO: use col_derive?
    # TODO: can I use the geodesic path for distance? Don't compute twice!

    def _func(t):
        if t - t0 < 1e-3:
            return float("inf")

        return func(t) - image

    return fsolve(_func, x0, xtol=xtol, maxfev=maxfev)


def get_sc_kwargs():
    sc_kwargs = {
        "marker": ".",
        "alpha": .5,
        "s": 20,
        "edgecolors": None,
        "linewidth": 0.
    }

    # "c": labels,
    # "cmap": "tab10",

    return sc_kwargs


def get_saving_kwargs():
    kwargs = {
        "format": "png",
        "pad_inches": 0,
        "dpi": 200
    }

    return kwargs


def get_qv_kwargs(grid_steps, x_min, x_max, y_min, y_max):
    qv_kwargs = {
        "angles": 'xy',
        "scale_units": 'xy',
        "scale": 2 * grid_steps / max(x_max - x_min, y_max - y_min),
        "headwidth": 2,
        "headlength": 2,
        "width": 0.005,
        "color": "navy"
    }

    return qv_kwargs


def get_coordinates(latent_activations, grid=None, num_steps=20, coords0=None, model_name=None):
    if grid == "dataset":
        coordinates = latent_activations
    else:

        # create grid
        x_min = torch.min(latent_activations[:, 0]).item()
        x_max = torch.max(latent_activations[:, 0]).item()
        y_min = torch.min(latent_activations[:, 1]).item()
        y_max = torch.max(latent_activations[:, 1]).item()

        num_steps_x = num_steps
        num_steps_y = ceil((y_max - y_min) / (x_max - x_min) * num_steps_x)

        step_size_x = (x_max - x_min) / (num_steps_x)
        step_size_y = (y_max - y_min) / (num_steps_y)

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

            # x_test = torch.linspace(x_min, x_max, steps=num_steps_x)
            # print(x)
            # print(x_test)
            # print(x.dtype)
            # print(x_test.dtype)
            # y = torch.linspace(y_min, y_max, steps=num_steps_y)
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

                if model_name is None:
                    factor = 0.3
                elif model_name == "ParametricUMAP":
                    factor = 0.95
                else:
                    factor = 0.3

                required_data_per_tile = factor * mean_data_per_tile

                if (i == 0 or i == num_xs - 1) or (j == 0 or j == num_ys - 1):
                    if (i == 0 or i == num_xs - 1) and (j == 0 or j == num_ys - 1):
                        required_data_per_tile = required_data_per_tile / 4
                    else:
                        required_data_per_tile = required_data_per_tile / 2

                if in_tile >= required_data_per_tile:
                    coordinates.append(torch.tensor([x, y]))

                # if in_tile < 0.3 * mean_data_per_tile:
                #    xs = torch.cat([xs[0:i], xs[i + 1:]])
                #    ys = torch.cat([ys[0:j], ys[j + 1:]])
                #    # print(in_tile, mean_data_per_tile)
                # else:
                #    coordinates.append(torch.tensor([x, y]))

        coordinates = torch.stack(coordinates)

        # else:
        #    num_steps_y = num_steps
        #    num_steps_x = int((x_max - x_min)/(y_max - y_min) * num_steps_y)

        # generate coordinate grid
        # grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        # prepare coordinate tuples
        # coordinates = torch.vstack([grid_x.ravel(), grid_y.ravel()]).T

        if grid == "min_square":
            pass
        elif grid == "convex_hull":
            hull = get_hull(latent_activations)
            coordinates = coordinates[in_hull(coordinates, hull)]
        else:
            coordinates = grid

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


def chunker(ar, size):
    return (ar[pos:pos + size] for pos in range(0, len(ar), size))


def minmax(item):
    return (item) / (torch.max(item) - torch.min(item)) + (torch.max(item) - torch.min(item)) / 2


def cmap_labels(labels, cmap=cm.turbo):
    # apply cmap and change base
    new_labels = (cmap(labels) * 255).astype(int)

    # remove opacity channel from rgba
    new_labels = torch.tensor(new_labels[:, :-1])

    return new_labels


def values_in_quantile(x, q=0):
    if q == 1.:
        idx = torch.arange(len(x))
    else:
        largest_abs = torch.topk(torch.abs(x), k=int(q * len(x)), largest=True)
        smallest = torch.topk(largest_abs.values, k=int(len(largest_abs.values) / len(x) * q * len(largest_abs.values)),
                              largest=False)

        idx = largest_abs.indices[smallest.indices]

    return idx


def get_new_percent(i, total, stepsize):
    return np.floor(i * 100 / total / stepsize) * stepsize


def determine_scaling_fn(scaling):
    # TODO: return auch direkt das prefix. Also if lin, return "" else scaling

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


def join_plots(subplots, latent_activations=None, labels=None, title="", cmap="tab10"):
    """

    """
    # add a plot of the latent space
    if latent_activations is not None:
        fig, axs = plt.subplots(1, len(subplots[0]) + 1)

        latent_activations = latent_activations.detach().cpu()
        sc_kwargs = get_sc_kwargs()

        # allocate axis
        if len(axs.shape) > 1:
            axis = axs[0, 0]
        else:
            axis = axs[0]

        axis.scatter(latent_activations[:, 0], latent_activations[:, 1], **sc_kwargs, c=labels, cmap=cmap)
        axis.set_aspect("equal")
        # axis.set_title("latent space")
    else:
        fig, axs = plt.subplots(1, len(subplots[0]))

    # fig.suptitle(title)

    # add axes
    for i, row in enumerate(subplots):
        for j, column in enumerate(row):
            obj = subplots[i][j]

            # skip first column if latent activations are passed, because they should be plotted there
            if latent_activations is not None:
                j_row = j + 1
            else:
                j_row = j

            # allocate axis
            if len(axs.shape) > 1:
                axis = axs[i, j_row]
            else:
                axis = axs[j_row]

            # unpack object containing subplot and possibly a collection
            if type(obj) == tuple:
                subplot, collection = obj
            else:
                subplot = obj
                collection = None

            position = axis.get_position()
            axis.remove()

            subplot.figure = fig
            fig.axes.append(subplot)
            fig.add_axes(subplot)

            subplot.set_position(position)

            if type(obj) == tuple:
                divider = make_axes_locatable(subplot)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(collection, cax=cax)

    return fig


def sample_points(rep1=None, rep2=None, num_samples=1):
    """
    Sample num_samples from rep1, and get the elements at the same position in rep2
    """

    # randomly sample the points
    rand_choice = torch.randperm(rep1.shape[0])[:num_samples]

    samples_rep1 = rep1[rand_choice, :]

    if rep2 is None:
        return samples_rep1

    samples_rep2 = rep2[rand_choice, :]

    return samples_rep1, samples_rep2


def distances(rep1=None, rep2=None, num_samples=1):
    """
    Compute distances in latent space and output space for a sample
    :param rep1:
    :param rep2: the outputs of the batch
    :param num_samples:
    :return: distances
    """

    # sample some points
    samples_rep1, samples_rep2 = sample_points(rep1=rep1, rep2=rep2, num_samples=num_samples)

    # compute the distances of the image of the origin to all the other points in the output space
    dist_rep2 = torch.squeeze(torch.cdist(rep2, samples_rep2, p=2))
    dist_rep1 = torch.squeeze(torch.cdist(rep1, samples_rep1, p=2))

    # set zero-distances very small
    dist_rep2 = torch.maximum(dist_rep2, LOWER_EPSILON * torch.ones_like(dist_rep2))
    dist_rep1 = torch.maximum(dist_rep1, LOWER_EPSILON * torch.ones_like(dist_rep1))

    # normalize distances
    dist_rep2_norm = dist_rep2 / torch.max(dist_rep2, dim=0).values
    dist_rep1_norm = dist_rep1 / torch.max(dist_rep1, dim=0).values

    return samples_rep1, samples_rep2, dist_rep1, dist_rep2, dist_rep1_norm, dist_rep2_norm


def prefix_to_filename(path):
    path_list = path.split("/")
    filename, extension = path_list[-1].split(".")


# from https://github.com/sdamrich/vis_utils/blob/5d57c0aa1b09e3a74dac48d89ae10031c1ed3b23/vis_utils/plot.py#L434

class Animator:
    def __init__(self, dim=2, ground_truth=None):
        self.data = []
        self.colors = []

        self.dim = dim

        self.ground_truth = ground_truth

        if ground_truth is not None:
            cmap = cm.get_cmap("twilight", 12)

            self.ground_truth_colors = torch.tensor(cmap(torch.zeros(ground_truth.shape[0])))[:, :-1]

        self.counter = 1

        self.x_max = 0
        self.x_min = 0
        self.y_max = 0
        self.y_min = 0
        self.z_min = 0
        self.z_max = 0

        self.scalebar = False

        # set up figure
        self.fig = plt.figure(constrained_layout=True)
        if self.dim == 2:
            self.ax = self.fig.gca()
        elif self.dim == 3:
            self.ax = self.fig.gca(projection="3d")

    def append(self, update, colors=None):
        # TODO: allow not passing rgb, then just zeros

        if colors is None:
            colors = torch.ones(update.shape[0])

        if self.ground_truth is not None:
            update = torch.cat((update, self.ground_truth))
            colors = torch.cat((colors, self.ground_truth_colors))
            # ground_truth_colors = torch.stack((torch.zeros(self.ground_truth.shape[0]), torch.zeros(self.ground_truth.shape[0]))).T

        self.data.append(update)

        self.colors.append(colors)

    def evaluate(self):
        self.data = torch.stack(self.data)
        self.colors = torch.stack(self.colors)

        self.x_max = torch.max(self.data[:, :, 0]).item()
        self.x_min = torch.min(self.data[:, :, 0]).item()
        self.y_max = torch.max(self.data[:, :, 1]).item()
        self.y_min = torch.min(self.data[:, :, 1]).item()

        if self.dim == 3:
            self.z_max = torch.max(self.data[:, :, 2]).item()
            self.z_min = torch.min(self.data[:, :, 2]).item()

        # self.labels = torch.ones(self.data.shape[1])

    def pad_data(self, n):
        """
        Copy first and last frame s.t. they are visible for one second
        """

        # copy first frame n times at the beginning
        self.data = torch.cat((self.data[0].expand(n, *self.data[0].shape), self.data[1:]), dim=0)
        self.colors = torch.cat((self.colors[0].expand(n, *self.colors[0].shape), self.colors[1:]), dim=0)

        # copy last frame n times at the end
        self.data = torch.cat((self.data[:-1], self.data[-1].expand(n, *self.data[-1].shape)), dim=0)
        self.colors = torch.cat((self.colors[:-1], self.colors[-1].expand(n, *self.colors[-1].shape)), dim=0)

    def minmax(self, data):
        scale = data.max() - data.min()
        data_scaled = (data - data.min()) / scale

        return data_scaled

    def get_scale(self, embd, max_length=0.5):
        # returns the smallest power of 10 that is smaller than max_length * the
        # spread in the x direction
        spreads = embd[:, 0].max() - embd[:, 0].min()
        spread = spreads.max()

        return 10 ** (int(np.log10(spread * max_length)))

    def get_current_angle(self):
        return self.counter * 360 / self.data.shape[0]

    def update_plot(self, data, scatter, scalebar=None):
        """
        Updates a scatter plot to create a video
        :param data: np.array (n ,2) Embedding information for next frame
        :param scatter: Scatter object
        :param scalebar: Scalebar object
        :return: scatter object and scalebar
        """

        # scale data according to minmax
        # data_scaled = self.minmax(data)
        data_scaled = data

        # update data
        if self.dim == 2:
            scatter.set_offsets(data_scaled)
            # scatter.set_data(data_scaled)

            self.ax.set_xlim([self.x_min, self.x_max])
            self.ax.set_ylim([self.y_min, self.y_max])

            # if self.dim == 3:
            #    self.ax.set_zlim([self.z_min, self.z_max])

            # update scalebar
            scale = data.max() - data.min()
            scalebar_length = self.get_scale(data)
            scalebar.size_bar.get_children()[0].set_width(scalebar_length / scale)
            scalebar.txt_label.set_text(str(scalebar_length))

        elif self.dim == 3:
            # scatter._offset3d = (data_scaled[:, 0], data_scaled[:, 1], data_scaled[:, 2])
            scatter._offsets3d = juggle_axes(data_scaled[:, 0].numpy(),
                                             data_scaled[:, 1].numpy(),
                                             data_scaled[:, 2].numpy(),
                                             "z")

            # self.ax.view_init(30, self.get_current_angle())

        # self.ax.set_xlim(data_scaled[:, 0].min(), data_scaled[:, 1].max())
        # self.ax.set_ylim(data_scaled[:, 1].min(), data_scaled[:, 1].max())
        # self.ax.set_aspect("equal")

        # update colors
        scatter.set_color(self.colors[self.counter].numpy())

        scatter.set_alpha(0.5)

        self.counter += 1

        if self.dim == 2:
            return scatter, scalebar,
        else:
            return scatter,

    def lim(self, lo, hi, eps=0.025):
        """
        Helper function for setting axis limits
        :param lo: float Min value
        :param hi: float Max value
        :param eps: float Limit values will by eps times lower and higher than the min and max value
        :return: tuple(float, float) Limit values
        """
        l = abs(hi - lo)
        return lo - l * eps, hi + l * eps

    def animate(self,
                filename,
                legend=False,
                scalebar=False,
                cmap="tab10",
                figsize=(4, 4),
                dpi=200,
                s=1,
                alpha=0.5,
                lim_eps=0.025,
                fps=50,
                **kwargs):
        """
        Create as video from a stack of 2D embeddings.
        :param data: np.array (t, n, 2) Stack of 2D embeddings
        :param labels: np.array (n) Labels of the embedding
        :param filename: str Output file name
        :param cmap: Matplotlib color map
        :param s: float Size of embedding points
        :param alpha: float Transparency
        :param lim_eps: float Excess size of the figure
        :param fps: int Frames per second
        :param kwargs: Additional arguments to plt.subplots
        :return: None
        """
        with plt.rc_context(fname=matplotlib.matplotlib_fname()):
            self.legend = legend

            self.fig.set_size_inches(*figsize)
            self.fig.set_dpi(dpi)

            # self.ax.set_xlim(*self.lim(0, 1, lim_eps))
            # self.ax.set_ylim(*self.lim(0, 1, lim_eps))

            # pad data
            self.pad_data(fps)

            # make initial plot
            # init_data = self.minmax(self.data[0])
            init_data = self.data[0]

            scatter = self.ax.scatter(
                *init_data.T, c=self.colors[0], marker=".", alpha=.5, s=20, edgecolors=None, linewidth=0.  # , cmap=cmap
            )

            if self.dim == 2:
                self.ax.set_axis_off()
                self.ax.set_aspect("equal")
            elif self.dim == 3:
                transform_axes(self.ax)

            if self.dim == 2:
                self.ax.set_xlim([self.x_min, self.x_max])
                self.ax.set_ylim([self.y_min, self.y_max])

            # if self.dim == 3:
            #    x_range = self.x_max - self.x_min
            #    y_range = self.y_
            #    self.ax.set_zlim([self.z_min, self.z_max])

            self.scalebar = scalebar

            # if legend:
            #    self.ax.legend(*scatter.legend_elements(), title="labels", loc="center left", bbox_to_anchor=(1, 0.5))

            if self.dim == 2:
                init_scale = self.data[0].max() - self.data[0].min()
                scalebar_length = self.get_scale(self.data[0])
                scalebar = AnchoredSizeBar(self.ax.transData,
                                           scalebar_length / init_scale,
                                           str(scalebar_length),
                                           loc="lower right",
                                           frameon=False)

                if self.scalebar:
                    scalebar = self.ax.add_artist(scalebar)

            # self.fig.savefig("/export/home/pnazari/workspace/AutoEncoderVisualization/stuff/test.png")

            # create animation object
            if self.dim == 2:
                sc_ani = animation.FuncAnimation(
                    self.fig,
                    self.update_plot,
                    frames=self.data[1:],
                    fargs=(scatter, scalebar),
                    interval=fps,
                    blit=True,
                    init_func=lambda: [scatter, scalebar],
                    save_count=len(self.data),
                )
            else:
                sc_ani = animation.FuncAnimation(
                    self.fig,
                    self.update_plot,
                    frames=self.data[1:],
                    fargs=(scatter,),
                    interval=fps,
                    blit=True,
                    init_func=lambda: [scatter],
                    save_count=len(self.data),
                )

            plt.show()

            # save animation
            sc_ani.save(
                str(filename),
                fps=fps,
                metadata={"artist": "Anonymous"},
                savefig_kwargs={"transparent": True},
            )


# problem with torch.mode: if there is a draw, it returns the minimal input. This bias is stupid
def get_max_vote(input):
    bincount = torch.bincount(input)

    # increase speed: https://discuss.pytorch.org/t/count-number-occurrence-of-value-per-row/137061/2
    max_votes = torch.where(bincount == bincount.max())

    max_vote = max_votes[0][torch.randperm(max_votes[0].size(0))[0]].item()

    return max_vote


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


def get_rg_value(data):
    # sm = matplotlib.cm.ScalarMappable()
    # rgb = torch.tensor(sm.to_rgba(torch.norm(data, dim=1))[:, :-1])
    # return rgb
    # print(rgb.shape)

    data_normed = torch.zeros_like(data)

    data_normed[:, 0] = minmax(data[:, 0])
    data_normed[:, 1] = minmax(data[:, 1])

    rgb = torch.vstack((0.5 * torch.ones(data_normed.shape[0]),
                        data_normed[:, 0],
                        data_normed[:, 1])).T

    return rgb


# scatter plots
def get_scale(embd, max_length=0.5):
    # returns the smallest power of 10 that is smaller than max_length * the
    # spread in the x direction
    spreads = embd.max(0) - embd.min(0)
    spread = spreads.max()

    return 10 ** (int(np.log10(spread * max_length)))


def add_scale(ax, embd):
    """
    Adds a scale bar
    """
    scale = get_scale(embd)
    embd_w, embd_h = embd.max(0) - embd.min(0)
    height = 0.005 * embd_h

    scalebar = AnchoredSizeBar(ax.transData,
                               scale,
                               str(scale),
                               loc="lower right",
                               size_vertical=height,
                               # borderpad= 0.5,
                               sep=4,
                               frameon=False,
                               fontproperties={"size": 20})
    ax.add_artist(scalebar)
    return scalebar


def plot_scatter(ax, x, y=None, title=None, cmap="tab10", s=1.0, alpha=0.5):
    """
    Produces a scatterplot for embeddings
    :param ax: Matplotlib axes to which the scatter plot is added
    :param x: np.array (n, 2) Embedding positions
    :param y: np.array (n) Label informations
    :param title: Title of the plot
    :return: Matplotlib axes
    """

    ax.scatter(*x.T, c=y, s=s, alpha=alpha, cmap=cmap, edgecolor="none")
    add_scale(ax, x)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


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
