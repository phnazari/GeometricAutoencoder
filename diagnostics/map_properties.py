import os
import warnings

import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data.handle_data import data_forward
from util import get_sc_kwargs, distances

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from conf import device


def circular_variance(test_loader, model, writer=None, output_path=None):
    """
    Calculate the variance in output-space-distance in donuts around smapled points
    :param test_loader: dataloader for the test data
    :param model: the model under consideration
    :param writer: SummaryWriter object
    :return:
    """
    print("[Analyse] Circular Variance ...")

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model.to(device), test_loader)

    num_rows = 2
    num_rings = 10

    # create other plots
    n_dist_samples = num_rows ** 2

    pimg_origin, _, dist_pimg, dist_img, _, dist_img_norm = distances(rep1=latent_activations,
                                                                      rep2=outputs,
                                                                      num_samples=n_dist_samples)

    warnings.filterwarnings("ignore")

    step_size = torch.max(dist_pimg).item() / num_rings

    # save the variances inside the rings and where the rings start
    variances = torch.zeros(num_rings, dist_img_norm.shape[1])
    dist_variances = torch.zeros(num_rings)

    for i in range(1, num_rings + 1):
        dist_variances[i - 1] = i * step_size
        for j in range(dist_img_norm.shape[1]):
            distance_mask = (((i - 1) * step_size < dist_pimg[:, j]) & (dist_pimg[:, j] < i * step_size))
            var = torch.var(dist_img_norm[:, j][distance_mask])
            variances[i - 1, j] = var

    mean_variances = torch.mean(variances, dim=1)
    var_variances = torch.var(variances, dim=1)

    """
    PLOTTING
    """

    dist_variances = dist_variances.detach().cpu()
    mean_variances = mean_variances.detach().cpu()
    var_variances = var_variances.detach().cpu()
    latent_activations = latent_activations.detach().cpu()
    pimg_origin = pimg_origin.detach().cpu()
    dist_img = dist_img.detach().cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    # fig.suptitle("Variance of distances inside Rings")

    ax.plot(dist_variances, mean_variances, c="navy", alpha=.5)
    ax.set_xlabel("distance in latent space")
    ax.set_ylabel(r"variance of distance in output space")
    ax.spines[['right', 'top']].set_visible(False)

    writer.add_figure("circular/statistics", plt.gcf())

    # create circle plots
    fig, axs = plt.subplots(num_rows, num_rows, figsize=(5, 5))
    # fig.suptitle("Distances in Latent Space")
    counter = 0

    for i in range(num_rows):
        for j in range(num_rows):

            # determine color scale depending
            if num_rows > 1:
                color = dist_img[:, counter]
                print("\n")
            else:
                color = dist_img

            scatter2 = axs[i, j].scatter(latent_activations[:, 0],
                                         latent_activations[:, 1],
                                         c=color,
                                         vmin=torch.min(dist_img),
                                         vmax=torch.max(dist_img),
                                         **get_sc_kwargs())

            if counter == 0:
                # divider = make_axes_locatable(axs[i, j])
                # cax = divider.append_axes("left", size="5%", pad=0.05)
                # plt.colorbar(scatter2, cax=cax)
                # cax.yaxis.set_ticks_position('left')

                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes("right", size="10%", pad=0.05)
                plt.colorbar(scatter2, cax=cax)
                cax.yaxis.set_ticks_position('right')

            else:
                axs[i, j].set_yticks([], [])
                axs[i, j].set_xticks([], [])
                axs[i, j].axis("off")

            # mark the origin
            axs[i, j].scatter(pimg_origin[counter][0].cpu(), pimg_origin[counter][1].cpu(), marker="2", s=30, c="red",
                              edgecolors=None)

            axs[i, j].set_aspect("equal")
            axs[i, j].set_xlim([torch.min(latent_activations[:, 0]).item(), torch.max(latent_activations[:, 0]).item()])
            axs[i, j].set_ylim([torch.min(latent_activations[:, 1]).item(), torch.max(latent_activations[:, 1]).item()])

            for k in range(dist_variances.shape[0]):
                circle = plt.Circle(pimg_origin[counter], dist_variances[k].item(), color="green", fill=False)
                axs[i, j].add_patch(circle)

            counter += 1

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    writer.add_figure("circular/circles", plt.gcf())
    # clean up tensorboard writer
    writer.flush()
    writer.close()


def distances_from_points(model,
                          dataloader,
                          num_rows=4,
                          num_cols=None,
                          output_path=None,
                          writer=None):
    """
    Calculate the distances from multiple sampled points in the output space and create grid plot
    :param dataloader: dataloader for the test data
    :param model: the model under consideration
    :param writer: summarywriter object
    :return:
    """

    print("[Analyse] distances from points ...")

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model.to(device), dataloader)

    # create other plots
    if num_cols is None:
        num_cols = num_rows

    n_dist_samples = num_cols * num_rows
    pimg_origin, img_origin, dist_pimg, dist_img, dist_pimg_norm, dist_img_norm = distances(rep1=latent_activations,
                                                                                            rep2=outputs,
                                                                                            num_samples=n_dist_samples)

    """
    PLOTTING
    """

    latent_activations = latent_activations.detach().cpu()
    outputs = outputs.detach().cpu()
    pimg_origin = pimg_origin.detach().cpu()
    dist_img = dist_img.detach().cpu()

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5, 5))
    # fig.suptitle("Distances in output space")

    num_rings = 10

    dist_variances = torch.zeros(num_rings)
    step_size = torch.max(dist_pimg).item() / num_rings

    for i in range(1, num_rings + 1):
        dist_variances[i - 1] = i * step_size

    counter = 0
    for i in range(num_rows):
        for j in range(num_cols):

            if num_rows > 1:
                if num_cols > 1:
                    axis = axs[i, j]
                else:
                    axis = axs[i]
            else:
                if num_cols > 1:
                    axis = axs[j]
                else:
                    axis = axs

            # determine color scale depending
            if num_rows > 1 or num_cols > 1:
                color = dist_img[:, counter]
            else:
                color = dist_img

            # color = color.view(-1, model.input_dim)

            scatter2 = axis.scatter(latent_activations[:, 0],
                                    latent_activations[:, 1],
                                    c=color,
                                    vmin=torch.min(dist_img),
                                    vmax=torch.max(dist_img),
                                    **get_sc_kwargs())

            if counter == 0:
                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(scatter2, cax=cax)
                cax.yaxis.set_ticks_position('right')

            axis.set_yticks([], [])
            axis.set_xticks([], [])
            axis.axis("off")

            # mark the origin
            axis.scatter(pimg_origin[counter, 0], pimg_origin[counter, 1],
                         marker="2",
                         s=30,
                         c="red",
                         edgecolors=None)

            # adjust axis
            axis.set_aspect("equal")
            axis.set_xlim([torch.min(latent_activations[:, 0]).item(), torch.max(latent_activations[:, 0]).item()])
            axis.set_ylim([torch.min(latent_activations[:, 1]).item(), torch.max(latent_activations[:, 1]).item()])

            for k in range(dist_variances.shape[0]):
                circle = plt.Circle(pimg_origin[counter], dist_variances[k].item(), color="green", fill=False)
                axis.add_patch(circle)

            counter += 1

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("distances/multiple", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()
