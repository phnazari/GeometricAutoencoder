import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data.handle_data import data_forward
from util import get_coordinates, get_sc_kwargs, determine_scaling_fn, values_in_quantile, join_plots, transform_axes


def decoder_knn_variance_latent(model,
                                dataloader,
                                k=30,
                                scaling="lin",
                                quantile=1.,
                                output_path=None,
                                writer=None,
                                **kwargs):
    print("[Analysis] decoder knn variance...")

    # TODO problem here: std scales linearly when space scales

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    # pairwise distances
    distances_output = torch.cdist(outputs, outputs)
    # knns in both spaces
    knn_output = torch.topk(distances_output, k=k, largest=False, dim=0)

    # preimages of the knns (so now lying in latent space)

    preimage_knn_output = (latent_activations[knn_output.indices] - latent_activations).pow(2).sum(2).sqrt()

    # var_preimage_knn_output = torch.var(latent_activations[knn_output.indices], dim=0)

    # variance of the preimages
    var_preimage_knn_output = torch.mean(preimage_knn_output, dim=0)
    var_preimage_knn_output /= torch.std(latent_activations)

    # scale variances
    scaling_fn, prefix = determine_scaling_fn(scaling)
    var_preimage_knn_output = scaling_fn(var_preimage_knn_output)

    # determine curvature values that lie in quantile
    idx_preimage = values_in_quantile(var_preimage_knn_output, quantile)

    """
    PLOTTING
    """

    latent_activations = latent_activations.detach().cpu()
    var_preimage_knn_output = var_preimage_knn_output.detach().cpu()
    outputs = outputs.detach().cpu()

    # plot the first version: the variance of the natural representation
    fig, ax = plt.subplots(figsize=(5, 5))
    if model.latent_dim > 1:
        scatter = ax.scatter(latent_activations[:, 0][idx_preimage],
                             latent_activations[:, 1][idx_preimage],
                             c=var_preimage_knn_output[idx_preimage],
                             **get_sc_kwargs(),
                             **kwargs)
    else:
        scatter = ax.scatter(outputs[:, 0][idx_preimage],
                             outputs[:, 1][idx_preimage],
                             c=var_preimage_knn_output[idx_preimage],
                             **get_sc_kwargs(),
                             **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(scatter, cax=cax)

    cb.outline.set_visible(False)
    ax.axis("off")

    # fig.suptitle(rf"Decoder: {prefix}Variance in Latent-Space [$V_k$] ")

    ax.set_aspect("equal")
    ax.set_yticks([], [])
    ax.set_xticks([], [])

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("decoder knn variance", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()


def encoder_knn_variance_latent(model,
                                dataloader,
                                scaling="lin",
                                quantile=1.,
                                output_path=None,
                                output_path_3d=None,
                                k=30,
                                writer=None,
                                **kwargs):
    print("[Analysis] encoder knn variance...")

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    # distances in input space
    distances_input = torch.cdist(inputs, inputs)

    # knn of each data in input space
    knn_input = torch.topk(distances_input, k=k, largest=False, dim=0)

    # image of those knn, now lying in latent space
    image_knn_input = (latent_activations[knn_input.indices] - latent_activations).pow(2).sum(2).sqrt()

    # their variance
    var_image_knn_input = torch.mean(image_knn_input, dim=0)

    # normalize variance by the variance of the whole space, in order to gain somewhat of a scale-invariance
    var_image_knn_input /= torch.std(latent_activations)

    # scale variances
    scaling_fn, prefix = determine_scaling_fn(scaling)
    var_image_knn_input = scaling_fn(var_image_knn_input)

    # determine curvature values that lie in quantile
    idx_image = values_in_quantile(var_image_knn_input, quantile)

    """
    PLOTTING
    """

    latent_activations = latent_activations.detach().cpu()
    var_image_knn_input = var_image_knn_input.detach().cpu()
    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()

    # plot input variances
    fig, ax = plt.subplots(figsize=(5, 5))

    if model.latent_dim > 1:
        scatter = ax.scatter(latent_activations[:, 0][idx_image],
                             latent_activations[:, 1][idx_image],
                             c=var_image_knn_input[idx_image],
                             **get_sc_kwargs(),
                             **kwargs)
    else:
        scatter = ax.scatter(outputs[:, 0][idx_image],
                             outputs[:, 1][idx_image],
                             c=var_image_knn_input[idx_image],
                             **get_sc_kwargs(),
                             **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(scatter, cax=cax)

    # remove this if not paper
    cb.outline.set_visible(False)
    ax.axis("off")

    # fig.suptitle(fr"Encoder: {prefix}Variance in Latent-Space [$W_k$]")

    ax.set_aspect("equal")
    ax.set_yticks([], [])
    ax.set_xticks([], [])

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("encoder knn variance", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()

    if model.input_dim == 3 and output_path_3d is not None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(inputs[:, 0][idx_image], inputs[:, 1][idx_image], inputs[:, 2],
                             c=var_image_knn_input[idx_image])

        fig.colorbar(scatter, ax=ax)

        transform_axes(ax)

        plt.savefig(output_path_3d, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

        plt.show()

        if writer is not None:
            writer.add_figure("encoder knn variance/3d", fig)

            # clearn up tensorboard writer
            writer.flush()
            writer.close()


def encoder_gaussian_variance(model,
                              dataloader,
                              scaling="lin",
                              quantile=1.,
                              std=1.,
                              n_samples=10,
                              output_path=None,
                              writer=None):
    print("[Analysis] encoder gaussian variance...")

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    inputs_copied = inputs.unsqueeze(0).expand(n_samples, *inputs.shape)
    # inputs_copied = inputs.repeat(n_samples, 1, 1)
    samples = torch.normal(mean=inputs_copied, std=std)
    del inputs_copied

    image_samples = model.encoder(samples)

    dist_samples_image = (image_samples - latent_activations).pow(2).sum(dim=2).sqrt()
    var_dist_samples_image = torch.mean(dist_samples_image, dim=0)

    # scale variances
    scaling_fn, prefix = determine_scaling_fn(scaling)
    scaled_var_dist_samples_image = scaling_fn(var_dist_samples_image)

    # determine curvature values that lie in quantile
    middle_idx = values_in_quantile(scaled_var_dist_samples_image, quantile)

    """
    PLOTTING
    """

    latent_activations = latent_activations.detach().cpu()
    scaled_var_dist_samples_image = scaled_var_dist_samples_image.detach().cpu()
    outputs = outputs.detach().cpu()

    fig, ax = plt.subplots(figsize=(5, 5))

    if model.latent_dim > 1:
        scatter2 = ax.scatter(latent_activations[:, 0][middle_idx],
                              latent_activations[:, 1][middle_idx],
                              c=scaled_var_dist_samples_image[middle_idx],
                              **get_sc_kwargs())
    else:
        scatter2 = ax.scatter(outputs[:, 0][middle_idx],
                              outputs[:, 1][middle_idx],
                              c=scaled_var_dist_samples_image[middle_idx],
                              **get_sc_kwargs())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(scatter2, cax=cax)

    # remove if not paper
    cb.outline.set_visible(False)
    ax.axis("off")

    # fig.suptitle(fr"Encoder: {prefix}Gaussian Variance")

    ax.set_aspect("equal")
    ax.set_yticks([], [])
    ax.set_xticks([], [])

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("encoder gaussian variance", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()
