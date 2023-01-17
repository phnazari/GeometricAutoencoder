import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.diagnostics.metric_properties import plot_determinants, plot_indicatrices
from src.diagnostics.representations import plot_latent_space

from src.models.submodules import BoxAutoEncoder

from data.handle_data import load_data
from firelight.visualizers.colorization import get_distinct_colors
from matplotlib.colors import ListedColormap
from conf import device, output_path
from numpy import genfromtxt

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

""" modify """


def evaluate(writer_dir=None,
             model_path=None,
             img_path=None,
             dataset_name="MNIST",
             model_name="Vanilla",
             used_diagnostics=None):
    """
    Determine settings
    """

    if dataset_name == "Zilionis_normalized":
        dataset_name = "Zilionis"
    if dataset_name == "artificial":
        dataset_name = "Earth"

    # choose Autoencoder model
    AE = BoxAutoEncoder

    train_batch_size = 256

    # set input dimensions
    if dataset_name == "Earth":
        input_dim = 3
        latent_dim = 2
        input_dims = (1, 3)
        num_labels = 6
    elif dataset_name in ["MNIST", "FashionMNIST"]:
        input_dim = 784
        latent_dim = 2
        input_dims = (1, 28, 28)
        num_labels = 10
    elif dataset_name == "Zilionis":
        input_dim = 306
        latent_dim = 2
        input_dims = (1, 306)
        num_labels = 20
    elif dataset_name in ["PBMC", "PBMC_new"]:
        input_dim = 50
        latent_dim = 2
        input_dims = (1, 50)
        num_labels = 11
    elif dataset_name == "CElegans":
        input_dim = 100
        latent_dim = 2
        input_dims = (1, 100)
        num_labels = 37

    # Prepare SummaryWriter
    # writer = get_summary_writer(subdir=writer_dir)
    writer = None

    train_loader, test_loader = load_data(train_batch_size=train_batch_size,
                                          test_batch_size=256,
                                          dataset=dataset_name)

    # set and create path for saving model
    model_path_save = os.path.join(output_path,
                                   f"models/{dataset_name}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/{writer_dir}")
    Path(model_path_save).mkdir(parents=True, exist_ok=True)

    # set and create path for saving images

    image_save_path = os.path.join(output_path,
                                   f"graphics/{dataset_name}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/",
                                   img_path)
    Path(image_save_path).mkdir(parents=True, exist_ok=True)

    """
    Analyse Model
    """
    print("[model] analyze ...")

    # create colormap
    if dataset_name != "Earth":
        seed = 0
        np.random.seed(seed)
        colors = get_distinct_colors(num_labels)
        np.random.shuffle(colors)
        cmap = ListedColormap(colors)
    else:
        cmap = "tab10"

    if model_name == "ParametricUMAP":
        if dataset_name == "MNIST":
            num_steps = 15
        elif dataset_name == "FashionMNIST":
            num_steps = 10
        elif dataset_name == "CElegans":
            num_steps = 10
        elif dataset_name == "Zilionis":
            num_steps = 13
        else:
            num_steps = 10
    elif model_name == "Vanilla":
        if dataset_name == "PBMC":
            num_steps = 20
        elif dataset_name == "Zilionis":
            num_steps = 20
        elif dataset_name == "FashionMNIST":
            num_steps = 10
        elif dataset_name == "MNIST":
            num_steps = 12
        else:
            num_steps = 10
    else:
        num_steps = 10

    # number of steps for indicatrices
    if dataset_name == "Earth":
        num_steps = 10

    # elif dataset_name == "PBMC":
    #     #if model_name == "Vanilla":
    #     #    num_steps = 20
    #     #elif model_name == "FashionMNIST":
    #     #    num_steps = 15
    #     else:
    #         num_steps = 10
    # elif dataset_name == "Zilionis":
    #    #if model_name == "Vanilla":
    #    #    num_steps = 20
    #    else:
    #        num_steps = 10
    # else:
    #    num_steps = 10

    # number of angles for indicatrices
    num_gon = 500

    # create diagnostics
    # if input_dim == 3:
    #    # plot input dataset
    #    plot_dataset(model,
    #                 train_loader,
    #                 input_dim=input_dim,
    #                 writer=writer,
    #                 output_path=os.path.join(image_save_path, "dataset.png")
    #                 )

    if model_name in ["Vanilla", "TopoReg", "GeomReg", "ParametricUMAP"]:
        # create model
        print(f"[model] move to {device}...")
        model = AE(input_shape=input_dim, latent_dim=latent_dim, input_dims=input_dims).to(device)  # , input_dims=(3, )
        model.load(model_path)

        if "determinants" in used_diagnostics:
            # determinants
            plot_determinants(model, train_loader, quantile=.95, batch_size=500, scaling="log", device=device,
                              output_path=os.path.join(image_save_path, "det.png"), writer=writer,
                              model_name=model_name, dataset_name=dataset_name)

        if "indicatrices" in used_diagnostics:
            # calculate indicatrices
            plot_indicatrices(model,
                              train_loader,
                              device=device,
                              cmap=cmap,
                              num_steps=num_steps,
                              num_gon=num_gon,
                              model_name=model_name,
                              dataset_name=dataset_name,
                              output_path=os.path.join(image_save_path, "indicatrices.png"),
                              writer=writer)

        if "embedding" in used_diagnostics:
            # plot latent space
            plot_latent_space(model, train_loader, cmap=cmap, dataset_name=dataset_name,
                              output_path=os.path.join(image_save_path, "latents.png"), writer=writer)

    else:
        model = None
        latent_data = genfromtxt(os.path.join("/".join(model_path.split("/")[:-1]), 'train_latents.csv'),
                                 skip_header=1,
                                 delimiter=',')

        latent_activations = latent_data[:, [0, 1]]
        labels = latent_data[:, 2]

        latent_activations = torch.from_numpy(latent_activations)
        labels = torch.from_numpy(labels)

        if model_name == "PCA":
            if "indicatrices" in used_diagnostics:
                # calculate indicatrices
                plot_indicatrices(model,
                                  train_loader,
                                  device=device,
                                  cmap=cmap,
                                  num_steps=num_steps,
                                  num_gon=num_gon,

                                  dataset_name=dataset_name,
                                  output_path=os.path.join(image_save_path, "indicatrices.png"),
                                  writer=writer,
                                  model_name=model_name,
                                  latent_activations=latent_activations,
                                  labels=labels)

            if "determinants" in used_diagnostics:
                # determinants
                plot_determinants(model, train_loader, quantile=.97, batch_size=500, scaling="log", device=device,
                                  output_path=os.path.join(image_save_path, "det.png"), writer=writer,
                                  latent_activations=latent_activations, model_name=model_name,
                                  dataset_name=dataset_name)

        if "embedding" in used_diagnostics:
            # plot latent space

            # if model_name in ["TSNE", "UMAP"]:
            #     meta = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/raw/pbmc_new", "zheng17-cell-labels.txt"), sep="\t", header=None, skiprows=1)
            #    meta = meta.to_numpy()[:, 1]
            #    cell_types = np.squeeze(meta)

            #    labels = np.zeros(len(cell_types)).astype(int)
            #    for i, phase in enumerate(np.unique(cell_types)):
            #        labels[cell_types == phase] = i

            #    labels = torch.tensor(labels)

            plot_latent_space(model, train_loader, cmap=cmap, dataset_name=dataset_name,
                              output_path=os.path.join(image_save_path, "latents.png"), writer=writer,
                              latent_activations=latent_activations, labels=labels)
