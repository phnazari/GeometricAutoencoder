import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from conf import UPPER_EPSILON
from data.handle_data import data_forward

from diffgeo.manifolds import RiemannianManifold
from diffgeo.metrics import PullbackMetric
from diffgeo.connections import LeviCivitaConnection
from sklearn.decomposition import PCA

from util import (get_sc_kwargs,
                  get_coordinates,
                  values_in_quantile,
                  determine_scaling_fn,
                  get_saving_kwargs)


def plot_determinants(model,
                      dataloader,
                      quantile=1.,
                      batch_size=-1,
                      scaling="asinh",
                      grid="dataset",
                      num_steps=None,
                      device="cpu",
                      output_path_1=None,
                      output_path_2=None,
                      writer=None,
                      x_lim_hist=None,
                      latent_activations=None,
                      model_name="GeomReg",
                      dataset_name="MNIST"):
    print("[Analyse] determinants ...")

    # forward pass
    if latent_activations is None:
        _, _, latent_activations, _ = data_forward(model, dataloader)

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(latent_activations.shape[0], generator=generator)
    latent_activations = latent_activations[perm]

    # calculate coordinates
    coordinates = get_coordinates(latent_activations, grid=grid, num_steps=num_steps).to(device)

    if model_name != "PCA":
        # initialize diffgeo objects
        pbm = PullbackMetric(2, model.decoder)
        lcc = LeviCivitaConnection(2, pbm)
        rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

        # batch-size is negative use the whole batch, i.e. don't batch
        if batch_size == -1:
            batch_size = coordinates.shape[0]

        # calculate determinants
        determinants = None
        for coord in torch.utils.data.DataLoader(coordinates, batch_size=batch_size):
            batch_dets = rm.metric_det(base_point=coord).detach()

            if determinants is None:
                determinants = batch_dets
            else:
                determinants = torch.hstack((determinants, batch_dets))

        middle_idx = values_in_quantile(determinants, quantile)
        determinants_in_quantile = determinants[middle_idx]

        min_determinant_in_quantile = torch.min(determinants_in_quantile)
        max_determinant_in_quantile = torch.max(determinants_in_quantile)
        determinants[determinants < min_determinant_in_quantile] = min_determinant_in_quantile
        determinants[determinants > max_determinant_in_quantile] = max_determinant_in_quantile

    else:
        determinants = torch.ones(latent_activations.shape[0]) * 10

    # scale determinants
    scaling_fn, prefix = determine_scaling_fn(scaling)

    dets_scaled = scaling_fn(determinants)

    nonnan_idx = torch.argwhere(~torch.isnan(dets_scaled)).squeeze()

    dets_scaled = dets_scaled[nonnan_idx]
    coordinates = coordinates[nonnan_idx]

    dets_scaled[torch.isinf(dets_scaled)] = -44

    determinants_relative = dets_scaled / torch.mean(dets_scaled)

    dets_scaled = determinants_relative - 1

    """
    PLOTTING
    """

    coordinates = coordinates.detach().cpu()
    dets_scaled = dets_scaled.detach().cpu()

    # plot color-coded determinants
    fig, ax = plt.subplots(figsize=((5, 5)))

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.)
    plt.margins(0.01, 0.01)

    scatter = ax.scatter(coordinates[:, 0],
                         coordinates[:, 1],
                         c=dets_scaled,
                         cmap="turbo",
                         **get_sc_kwargs(),
                         vmin=-1.8, vmax=1.22)

    #if model_name == "GeomReg" and dataset_name == "MNIST":
    #    divider = make_axes_locatable(ax)
    #    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    #    sm = ScalarMappable()
    #    sm.set_cmap("turbo")
    #    sm.set_array(dets_scaled)  # determinants
    #    sm.set_clim(-1.8, 1.22)
    #    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    #    cbar.set_alpha(0.5)
    #    cbar.draw_all()

    ax.set_aspect("equal")
    ax.axis("off")

    if output_path_1 is not None:
        plt.savefig(output_path_1, **get_saving_kwargs())

    plt.show()

    if writer is not None:
        writer.add_figure("determinants/colorcode", fig)


def plot_indicatrices(model,
                      dataloader,
                      grid="convex_hull",
                      device="cpu",
                      num_steps=20,
                      num_gon=50,
                      output_path=None,
                      writer=None,
                      latent_activations=None,
                      model_name="GeomReg",
                      dataset_name="MNIST",
                      labels=None,
                      cmap="tab10",
                      inputs=None):
    print("[Analysis] Indicatrices...")

    if labels is None:
        passed = False
    else:
        passed = True

    if not passed:
        _, _, latent_activations, labels = data_forward(model, dataloader)
        latent_activations = latent_activations.detach().cpu()

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(labels.shape[0], generator=generator)
    latent_activations = latent_activations[perm]
    labels = labels[perm]

    idx0 = 45636
    coords0 = latent_activations[idx0]

    coords0 = None

    coordinates = get_coordinates(latent_activations, grid=grid, num_steps=num_steps, coords0=coords0,
                                  model_name=model_name).to(device)

    # calculate grid distance

    x_min = torch.min(latent_activations[:, 0]).item()
    x_max = torch.max(latent_activations[:, 0]).item()
    y_min = torch.min(latent_activations[:, 1]).item()
    y_max = torch.max(latent_activations[:, 1]).item()

    num_steps_x = num_steps
    num_steps_y = int((y_max - y_min) / (x_max - x_min) * num_steps_x)

    step_size_x = (x_max - x_min) / (num_steps_x)
    step_size_y = (y_max - y_min) / (num_steps_y)
    stepsize = min(step_size_x, step_size_y)

    # stepsize = min((x_max - x_min) / (num_steps - 1), (y_max - y_min) / (num_steps - 1)) * 2

    # find initial coordinate
    if coords0 is not None:
        coords0_index = None
        for i, row in enumerate(coordinates.cpu()):
            if torch.all(row.eq(coords0)):
                coords0_index = i

    if not passed:
        # initialize diffgeo objects
        pbm = PullbackMetric(2, model.decoder)
        lcc = LeviCivitaConnection(2, pbm)
        rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

        # generate vector patches at grid points, normed in pullback metric
        vector_patches, norm_vectors = rm.generate_unit_vectors(num_gon, coordinates)

        vector_patches = vector_patches.to(device)

        max_vector_norm = torch.min(torch.linalg.norm(vector_patches.view(-1, 2), dim=1))
    else:
        # the angles
        phi = torch.linspace(0., 2 * np.pi, num_gon)

        # generate circular vector patch
        raw_vectors = torch.stack([torch.sin(phi), torch.cos(phi)])

        normed_vectors = raw_vectors.to(device)
        vector_patches = 0 * coordinates[:, :, None] + normed_vectors[None, :, :]
        vector_patches = torch.transpose(vector_patches, 1, 2)
        max_vector_norm = 1

    if passed:
        scaling_factor = 1 / 4
    else:
        if model_name == "GeomReg":
            if dataset_name in ["MNIST", "FashionMNIST"]:
                scaling_factor = 1 / 5
            else:
                scaling_factor = 1 / 10
        elif model_name == "TopoReg" and dataset_name == "Zilionis":
            scaling_factor = 1 / 50
        else:
            scaling_factor = 1 / 10

    normed_vector_patches = vector_patches / max_vector_norm * stepsize * scaling_factor  # / 3
    anchored_vector_patches = coordinates.unsqueeze(1).expand(*normed_vector_patches.shape) + normed_vector_patches

    # create polygons
    polygons = [Polygon(tuple(vector.tolist()), True) for vector in anchored_vector_patches]

    if coords0 is not None:
        polygon0 = polygons.pop(coords0_index)

    """
    Plotting
    """
    latent_activations = latent_activations.detach().cpu()

    # plot blobs
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.)
    plt.margins(0.01, 0.01)

    ax.scatter(latent_activations[:, 0], latent_activations[:, 1], c=labels, cmap=cmap, **get_sc_kwargs())

    p = PatchCollection(polygons)

    if coords0 is None:
        p.set_facecolor([0 / 255, 0 / 255, 0 / 255, 0.5])
        p.set_edgecolor([0 / 255, 0 / 255, 0 / 255, 0.5])
    else:
        p.set_facecolor([0 / 255, 0 / 255, 0 / 255, 0.5])
        p.set_edgecolor([0 / 255, 0 / 255, 0 / 255, 0.5])
        polygon0.set_color([255 / 255, 0 / 255, 0 / 255, 0.5])
        ax.add_patch(polygon0)

    ax.add_collection(p)
    ax.set_aspect("equal")
    ax.axis("off")
    # fig.suptitle(f"Indicatrices")

    if output_path is not None:
        plt.savefig(output_path, **get_saving_kwargs())

    plt.show()

    if writer is not None:
        writer.add_figure("indicatrix", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()
