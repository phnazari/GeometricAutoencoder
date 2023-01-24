import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from data.handle_data import data_forward

from src.diffgeo.manifolds import RiemannianManifold
from src.diffgeo.metrics import PullbackMetric
from src.diffgeo.connections import LeviCivitaConnection

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
                      output_path=None,
                      writer=None,
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

    # get coordinates the determinants should be evaluated on
    coordinates = get_coordinates(latent_activations.detach().cpu(), grid=grid, num_steps=num_steps).to(device)

    if model_name != "PCA":
        # initialize diffgeo objects
        pbm = PullbackMetric(2, model.decoder)
        lcc = LeviCivitaConnection(2, pbm)
        rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

        # batch-size is negative use the whole batch, i.e. don't batch. Need to batch for storage reasons
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

        # collapse determinants into quantile
        middle_idx = values_in_quantile(determinants, quantile)
        determinants_in_quantile = determinants[middle_idx]
        min_determinant_in_quantile = torch.min(determinants_in_quantile)
        max_determinant_in_quantile = torch.max(determinants_in_quantile)
        determinants[determinants < min_determinant_in_quantile] = min_determinant_in_quantile
        determinants[determinants > max_determinant_in_quantile] = max_determinant_in_quantile

    else:
        determinants = torch.ones(latent_activations.shape[0]) * 10

    # scale determinants
    scaling = "log"
    scaling_fn, prefix = determine_scaling_fn(scaling)
    dets_scaled_raw = scaling_fn(determinants)

    # remove nan scaled determinants
    nonnan_idx = torch.argwhere(~torch.isnan(dets_scaled_raw)).squeeze()
    dets_scaled_raw = dets_scaled_raw[nonnan_idx]
    coordinates = coordinates[nonnan_idx]

    # dets_scaled[torch.isinf(dets_scaled)] = -44

    # change units and shift
    determinants_relative = dets_scaled_raw / torch.abs(torch.mean(dets_scaled_raw))
    # determinants_relative = dets_scaled_raw / torch.mean(dets_scaled_raw)
    dets_scaled = determinants_relative - 1

    print(dataset_name, torch.min(dets_scaled), torch.max(dets_scaled))

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

    if torch.mean(dets_scaled_raw) >= 0:
        cmap = "turbo"
    else:
        cmap = "turbo"
        dets_scaled += 2

    scatter = ax.scatter(coordinates[:, 0],
                         coordinates[:, 1],
                         c=dets_scaled,
                         cmap=cmap,
                         **get_sc_kwargs(),
                         vmin=-1.8,
                         vmax=1.22)

    # if model_name == "GeomReg" and dataset_name == "MNIST":
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("bottom", size="5%", pad=0.05)
    # sm = ScalarMappable()
    # sm.set_cmap("turbo")
    # sm.set_array(dets_scaled)  # determinants
    # sm.set_clim(-1.8, 1.22)
    # cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    # cbar.set_alpha(0.5)
    # cbar.draw_all()

    ax.set_aspect("equal")
    ax.axis("off")

    if output_path is not None:
        plt.savefig(output_path, **get_saving_kwargs())

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
                      cmap="tab10"):
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

    coordinates_on_data = get_coordinates(latent_activations,
                                          grid="on_data",
                                          num_steps=num_steps,
                                          coords0=coords0,
                                          dataset_name=dataset_name,
                                          model_name=model_name).to(device)

    coordinates_off_data = get_coordinates(latent_activations,
                                           grid="off_data",
                                           num_steps=num_steps,
                                           coords0=coords0,
                                           dataset_name=dataset_name,
                                           model_name=model_name).to(device)

    if model_name == "Vanilla" and dataset_name in ["Zilionis", "PBMC"]:
        coordinates = coordinates_on_data
    else:
        coordinates = torch.vstack([coordinates_on_data, coordinates_off_data])

    # coordinates = coordinates_on_data

    # calculate grid step sizes
    x_min = torch.min(latent_activations[:, 0]).item()
    x_max = torch.max(latent_activations[:, 0]).item()
    y_min = torch.min(latent_activations[:, 1]).item()
    y_max = torch.max(latent_activations[:, 1]).item()

    num_steps_x = num_steps
    num_steps_y = int((y_max - y_min) / (x_max - x_min) * num_steps_x)

    step_size_x = (x_max - x_min) / (num_steps_x)
    step_size_y = (y_max - y_min) / (num_steps_y)
    stepsize = min(step_size_x, step_size_y)

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

        vector_norms = torch.linalg.norm(vector_patches.reshape
                                         (-1, 2), dim=1)
        max_vector_norm = torch.min(vector_norms[torch.nonzero(vector_norms)])
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
            if dataset_name == "MNIST":
                scaling_factor = 1 / 5
            elif dataset_name == "FashionMNIST":
                scaling_factor = 1 / 7
            elif dataset_name == "Earth":
                scaling_factor = 1 / 5
            else:
                scaling_factor = 1 / 10
        elif model_name == "TopoReg":
            if dataset_name == "Zilionis":
                scaling_factor = 1 / 50
            elif dataset_name == "PBMC":
                scaling_factor = 1 / 30
            else:
                scaling_factor = 1 / 20
        elif model_name == "ParametricUMAP":
            if dataset_name == "PBMC":
                scaling_factor = 1 / 30
            elif dataset_name == "FashionMNIST":
                scaling_factor = 1 / 20
            else:
                scaling_factor = 1 / 15
        elif model_name == "Vanilla":
            if dataset_name == "PBMC":
                scaling_factor = 1 / 50
            elif dataset_name == "Zilionis":
                scaling_factor = 1 / 40
            elif dataset_name == "Earth":
                scaling_factor = 1 / 20
            elif dataset_name == "CElegans":
                scaling_factor = 1 / 70
            elif dataset_name == "MNIST":
                scaling_factor = 1 / 20
            else:
                scaling_factor = 1 / 50
        else:
            scaling_factor = 1 / 10

    normed_vector_patches = vector_patches / max_vector_norm * stepsize * scaling_factor  # / 3
    anchored_vector_patches = coordinates.unsqueeze(1).expand(*normed_vector_patches.shape) + normed_vector_patches

    # create polygons
    polygons = [Polygon(tuple(vector.tolist()), True) for vector in anchored_vector_patches]

    polygons_on_data = polygons[:coordinates_on_data.shape[0]]
    polygons_off_data = polygons[coordinates_on_data.shape[0]:]

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

    ax.scatter(latent_activations[:, 0],
               latent_activations[:, 1],
               c=labels,
               cmap=cmap,
               **get_sc_kwargs())

    p_on_data = PatchCollection(polygons_on_data)
    p_off_data = PatchCollection(polygons_off_data)
    # p2 = PatchCollection(polygons2)

    # p_off_data.set_edgecolor([0 / 255, 0 / 255, 0 / 255, 0.2])
    # if model_name == "Vanilla" and dataset_name == "Zilionis":
    #    p_off_data.set_facecolor([0 / 255, 0 / 255, 0 / 255, 0.0])
    # else:
    #    p_off_data.set_facecolor([0 / 255, 0 / 255, 0 / 255, 0.2])

    p_on_data.set_color([0 / 255, 0 / 255, 0 / 255, 0.3])
    p_off_data.set_color([0 / 255, 0 / 255, 0 / 255, 0.3])

    if coords0 is not None:
        polygon0.set_color([255 / 255, 0 / 255, 0 / 255, 0.2])
        ax.add_patch(polygon0)

    ax.add_collection(p_off_data)
    ax.add_collection(p_on_data)
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
