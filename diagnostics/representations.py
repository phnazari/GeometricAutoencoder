import os
import torch
from data.custom import Earth, Zilionis, PBMC, CElegans
from matplotlib import pyplot as plt

from data.handle_data import data_forward
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from torch import nn
from util import get_sc_kwargs, cmap_labels, transform_axes, pathpatch_2d_to_3d, pathpatch_translate, get_saving_kwargs


def plot_latent_space(model,
                      dataloader,
                      cmap="tab10",
                      dataset=None,
                      output_path=None,
                      writer=None,
                      latent_activations=None,
                      labels=None,
                      with_legend=False):
    print("[Analyse] latent representation...")

    if latent_activations is None:
        inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(labels.shape[0], generator=generator)
    latent_activations = latent_activations[perm]
    labels = labels[perm]

    with_legend = True

    """
    Plotting
    """

    latent_activations = latent_activations.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    if not with_legend:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.)
    plt.margins(0.01, 0.01)

    scatter = ax.scatter(latent_activations[:, 0],
                         latent_activations[:, 1],
                         c=labels,
                         cmap=cmap,
                         **get_sc_kwargs())

    if with_legend:
        if dataset == "Earth":
            handles, _ = scatter.legend_elements(num=None)

            chartBox = ax.get_position()
            ax.set_position([0, 0, chartBox.width, chartBox.height])

            # string_labels = Earth().transform_labels(labels)
            string_labels = ["Africa", "Europe", "Asia", "North America", "Australia", "South America"]
            ax.legend(handles, string_labels, loc="center left", bbox_to_anchor=(1, 0.5))
        elif dataset == "Zilionis":
            handles, _ = scatter.legend_elements(num=None)

            chartBox = ax.get_position()
            ax.set_position([0, 0, chartBox.width, chartBox.height])

            string_labels = Zilionis().transform_labels(
                os.path.join(os.path.dirname(__file__), '..', "data/raw/zilionis"))
            ax.legend(handles, string_labels, loc="center left", bbox_to_anchor=(1, 0.5))
        elif dataset == "PBMC":
            handles, _ = scatter.legend_elements(num=None)

            string_labels = PBMC().transform_labels(
                os.path.join(os.path.dirname(__file__), '..', "data/raw/pbmc"))
            ax.legend(handles, string_labels, loc="center left", bbox_to_anchor=(1, 0.5))
        elif dataset == "CElegans":
            handles, _ = scatter.legend_elements(num=None)

            string_labels = CElegans().transform_labels(
                os.path.join(os.path.dirname(__file__), '..', "data/raw/celegans"))

            chartBox = ax.get_position()
            ax.set_position([0, 0, chartBox.width, chartBox.height])

            ax.legend(handles, string_labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
            scatter.legend_elements(prop="colors", num=37)
        else:
            ax.legend(*scatter.legend_elements(num=None), loc="center left", bbox_to_anchor=(1, 0.5))

    if output_path is not None:
        plt.savefig(output_path, **get_saving_kwargs())

    plt.show()

    if writer is not None:
        writer.add_figure("latent space", fig)


def plot_dataset(model,
                 dataloader,
                 input_dim=None,
                 output_path=None,
                 writer=None):
    print("[Analyse] Dataset")

    if input_dim != 3:
        return

    inputs, _, _, labels = data_forward(model, dataloader)

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(labels.shape[0], generator=generator)
    inputs = inputs[perm]
    labels = labels[perm]

    idx0 = 45636
    coords0 = inputs[idx0]

    """
    PLOTTING
    """

    inputs = inputs.detach().cpu()
    labels = labels.detach().cpu()

    sc_kwargs = get_sc_kwargs()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.view_init(azim=90)

    ax.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=labels, cmap="tab10", s=10, marker=".", alpha=.4)

    p = Circle((0, 0), .3, color=[255 / 255, 0 / 255, 0 / 255, 1])  # Add a circle in the yz plane

    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=tuple(coords0.cpu()))
    pathpatch_translate(p, tuple(coords0.cpu()))

    transform_axes(ax)

    if output_path is not None:
        plt.savefig(output_path, **get_saving_kwargs())

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("dataset", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()


def plot_reconstruction(model,
                        dataloader,
                        input_dim=None,
                        output_path=None,
                        output_path_2=None,
                        writer=None):
    print("[Analyse] Reconstruction")

    # The MNIST case
    if input_dim == 784:
        inputs, outputs, latent_activations, labels = data_forward(model, dataloader)
        inputs = inputs[0].view(28, 28)
        outputs = outputs[0].view(28, 28)
    else:
        inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    """
    PLOTTING
    """

    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()
    latent_activations = latent_activations.detach().cpu()

    sc_kwargs = get_sc_kwargs()

    if input_dim == 784:
        fig_sum, (ax1, ax3) = plt.subplots(1, 2, figsize=(5, 5))

        ax1.imshow(inputs)
        ax1.set_aspect("equal")

        ax3.imshow(outputs)
        ax3.set_aspect("equal")
    elif input_dim == 3:
        fig_sum = plt.figure()
        ax1 = fig_sum.add_subplot(1, 3, 1, projection='3d')
        ax1.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=labels, cmap="viridis", s=10, marker=".", alpha=.4)
        ax1.view_init(azim=20)

        ax2 = fig_sum.add_subplot(1, 3, 2)
        ax2.scatter(latent_activations[:, 0], latent_activations[:, 1], **sc_kwargs, c=labels)
        ax2.set_yticks([], [])
        ax2.set_xticks([], [])
        ax2.set_aspect("equal")

        ax3 = fig_sum.add_subplot(1, 3, 3, projection="3d")
        ax3.scatter(outputs[:, 0], outputs[:, 1], outputs[:, 2], c=labels, cmap="viridis", s=10, marker=".", alpha=.4)
        ax3.view_init(azim=30)

        fig_recon = plt.figure()
        ax_recon = plt.axes(projection='3d')

        ax_recon.scatter(outputs[:, 0], outputs[:, 1], outputs[:, 2], c=labels, cmap="viridis", s=10, marker=".",
                         alpha=.4)

        ax_recon.view_init(azim=20)
        transform_axes(ax_recon)

        if output_path_2 is not None:
            plt.savefig(output_path_2, **get_saving_kwargs())

    elif input_dim == 2:
        fig_sum, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5, 5))

        ax1.scatter(inputs[:, 0], inputs[:, 1], c=labels, s=1)
        ax2.scatter(latent_activations, torch.zeros_like(latent_activations), c=labels, s=1, marker="v")
        ax3.scatter(outputs[:, 0], outputs[:, 1], c=labels, s=1)

        ax1.set_aspect("equal")
        ax3.set_aspect("equal")

    else:
        print("No Reconstruction")
        return

    if output_path is not None:
        plt.savefig(output_path, **get_saving_kwargs())

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("reconstruction", fig_sum)

        # clean up tensorboard writer
        writer.flush()
        writer.close()
