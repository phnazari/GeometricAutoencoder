import _pickle
import os

import pandas as pd
import torch

from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functorch import vmap

from data.handle_data import data_forward
from util import get_coordinates, get_sc_kwargs, determine_scaling_fn, values_in_quantile, join_plots, get_max_vote


def plot_knn_performance(model,
                         dataloader,
                         writer=None,
                         output_path=None,
                         k=30):
    """
    Evaluate knn classification in latent-space
    """

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    # pairwise distances
    distances_latent = torch.cdist(latent_activations, latent_activations)

    # knns in both spaces
    knn = torch.topk(distances_latent, k=k, largest=False, dim=1)

    # calculate knn classification success rate
    knn_matches = torch.eq(labels[knn.indices], labels.unsqueeze(1))
    knn_success_rate = torch.sum(knn_matches, dim=1) / k

    """
    Plotting
    """

    latent_activations = latent_activations.detach().cpu()
    outputs = outputs.detach().cpu()
    knn_success_rate = knn_success_rate.detach().cpu()

    # plot the first version: the variance of the natural representation
    fig_knn, ax_knn = plt.subplots(figsize=(5, 5))
    # fig_knn.suptitle("Nearest Neighbour Similarity")

    if model.latent_dim > 1:
        scatter_knn = ax_knn.scatter(latent_activations[:, 0],
                                     latent_activations[:, 1],
                                     c=knn_success_rate,
                                     **get_sc_kwargs())
    else:
        scatter_knn = ax_knn.scatter(outputs[:, 0],
                                     outputs[:, 1],
                                     c=knn_success_rate,
                                     **get_sc_kwargs())
    divider = make_axes_locatable(ax_knn)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(scatter_knn, cax=cax)

    cb.outline.set_visible(False)
    ax_knn.axis("off")

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("classification/similarity", fig_knn)

    ax_knn.remove()

    if model.latent_dim > 1:
        fig_both = join_plots([[(ax_knn, scatter_knn)]], latent_activations=latent_activations, labels=labels)
    else:
        fig_both = join_plots([[(ax_knn, scatter_knn)]], latent_activations=outputs, labels=labels)

    plt.show()

    if writer is not None:
        writer.add_figure("classification/similarity/both", fig_both)

        writer.flush()
        writer.close()


def get_classification_error(model, dataloader, model_paths, k, xranges):
    knn_errors = []
    mse_errors = []

    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    inputs = inputs.detach()
    outputs = outputs.detach()
    latent_activations = latent_activations.detach()

    labels = labels.type(torch.int32)

    distances = torch.cdist(inputs, inputs)
    knn = torch.topk(distances, k=k, largest=False, dim=1)
    knn_labels = labels[knn.indices]
    knn_vote = torch.tensor([get_max_vote(row) for row in knn_labels])

    # calculate error classification rate
    error_rate = 1 - torch.sum(knn_vote == labels) / len(labels)

    for i, model_path in enumerate(model_paths):
        knn_error = []
        mse_error = []

        # knn_errors = torch.zeros(num_models)
        # mse_error = torch.zeros(num_models)
        for j, file in enumerate(os.listdir(model_path)):
            print(f"model nr. {i + 1}, file nr. {j + 1}")
            # there might be a readme.txt
            try:
                # load statedict to model
                # print(file)
                # print(os.path.join(model_path, file))
                model.load(os.path.join(model_path, file))
            except (_pickle.UnpicklingError):
                continue

            # forward pass
            inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

            inputs = inputs.detach()
            outputs = outputs.detach()
            latent_activations = latent_activations.detach()

            labels = labels.type(torch.int32)

            # pairwise distances
            distances_latent = torch.cdist(latent_activations, latent_activations)

            # knns in both spaces
            knn = torch.topk(distances_latent, k=k, largest=False, dim=1)

            knn_labels = labels[knn.indices]

            # majority vote
            # knn_vote, knn_vote_indices = torch.mode(knn_labels, dim=1)
            knn_vote = torch.tensor([get_max_vote(row) for row in knn_labels])

            # calculate error classification rate
            error_rate = 1 - torch.sum(knn_vote == labels) / len(labels)

            # save error rates
            knn_error.append(error_rate)
            mse_error.append(nn.MSELoss()(inputs, outputs).item())

        # mse_error = torch.tensor(mse_error)
        knn_error = torch.tensor(knn_error)
        mse_error = torch.tensor(mse_error)

        xranges[i], order = torch.sort(xranges[i])

        knn_error = knn_error[order]
        knn_errors.append(knn_error.detach().cpu())
        mse_errors.append(mse_error)

        # mse_error = mse_error[order]
        # xranges[i] = xranges[i]

        test = torch.stack(knn_errors).squeeze(0)
        test2 = torch.stack(mse_errors).squeeze(0)

        #print(torch.mean(test))
        #print(torch.std(test))

        #print(torch.mean(test2))
        #print(torch.std(test2))

    return knn_errors, mse_errors


def classification_error_figure(model,
                                dataloader,
                                model_paths,
                                k=10,
                                xranges=None,
                                legend_labels=None,
                                writer=None,
                                output_path=None):
    knn_errors, mse_errors = get_classification_error(model, dataloader, model_paths, k, xranges)

    """
    PLOTTING
    """

    # TODO: errorbars for classification? Sowas wie #didnt vote for max / k

    # mse_error = mse_error.detach().cpu()
    # xranges = xranges.detach().cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    # fig.suptitle(f"Classification (k={k} in latent space)- and Reconstruction Loss")

    ax.set_xlabel("regularization")
    ax.set_ylabel("classification error")
    ax.set_xticks([0])

    # TODO: also show reconstruction error for all runs?

    for i in range(len(xranges)):
        ax.plot(knn_errors[i], label=legend_labels[i])

    # ax2 = ax1.twinx()
    # l2, = ax2.plot(xranges, mse_error, label="reconstruction loss", c="crimson")

    # plt.legend([l1, l2], ["classification loss", "reconstruction loss"])

    plt.legend()

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    if writer is not None:
        writer.add_figure("classification/loss/figure", fig)

        writer.flush()
        writer.close()


def classification_error_table(model,
                               dataloader,
                               model_paths,
                               k=10,
                               xranges=None,
                               legend_labels=None,
                               writer=None,
                               output_path=None):
    knn_errors, mse_errors = get_classification_error(model, dataloader, model_paths, k, xranges)

    best_knn = torch.zeros(len(model_paths) + 1)
    best_mse = torch.zeros(len(model_paths) + 1)
    baseline_knns = torch.zeros(len(model_paths))
    baseline_mses = torch.zeros(len(model_paths))
    for i in range(len(model_paths)):
        knn_error = knn_errors[i]
        mse_error = mse_errors[i]
        baseline_knns[i] = knn_error[0].item()
        baseline_mses[i] = mse_error[0].item()
        best_knn[i + 1] = torch.min(knn_error[1:]).item()
        best_mse[i + 1] = torch.min(mse_error[1:]).item()

    baseline_knn = torch.mean(baseline_knns)
    baseline_mse = torch.mean(baseline_mses)

    best_knn[0] = baseline_knn
    best_mse[0] = baseline_mse

    best_knn = torch.round(best_knn, decimals=3)
    best_mse = torch.round(best_mse, decimals=4)

    data = {
        r"$\mathbf{knn}$": best_knn,
        r"$\mathbf{mse}$": best_mse
    }

    df = pd.DataFrame(data, index=[r"$\mathbf{-}$", *legend_labels])

    """
    PLOTTING
    """

    # mse_error = mse_error.detach().cpu()
    # xranges = xranges.detach().cpu()

    fig = plt.figure()
    plt.axis('off')
    plt.axis('auto')
    # fig.suptitle(f"Classification (k={k} in latent space)- and Reconstruction Loss")

    table = plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center", cellLoc="center")

    table.set_fontsize(14)
    # pd.plotting.table(ax, df)

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    if writer is not None:
        writer.add_figure("classification/loss/table", fig)

        writer.flush()
        writer.close()
