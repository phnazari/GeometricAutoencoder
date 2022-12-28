import os
from datetime import datetime
from pathlib import Path
from decimal import Decimal

import conf
import numpy as np
import torch

from diagnostics.metric_properties import plot_determinants, plot_indicatrices
from diagnostics.representations import plot_latent_space, plot_dataset

from src.models.submodules import ELUUMAPAutoEncoder

from data.handle_data import load_data, data_forward
from firelight.visualizers.colorization import get_distinct_colors
from matplotlib.colors import ListedColormap
from conf import device, get_summary_writer, output_path
from numpy import genfromtxt

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

""" modify """


# currently used auto encoder
# AE = DeepThinAutoEncoder
# AE = ELUAutoEncoder
# AE = TestELUAutoEncoder
# AE = ThinAutoEncoder
# AE = WideShallowAutoEncoder
# AE = WideDeepAutoEncoder
# AE = TopoAutoEncoder
# AE = UMAPAutoEncoder


def evaluate(writer_dir=None,
             model_path=None,
             img_path=None,
             dataset="MNIST",
             model_name="Vanilla",
             used_diagnostics=None):
    """
    Determine settings
    """

    if dataset == "Zilionis_normalized":
        dataset = "Zilionis"
    if dataset == "artificial":
        dataset = "Earth"

    # choose Autoencoder model
    AE = ELUUMAPAutoEncoder

    train_batch_size = 256

    # set input dimensions
    if dataset == "Earth":
        input_dim = 3
        latent_dim = 2
        input_dims = (1, 3)
    elif dataset in ["MNIST", "FashionMNIST"]:
        input_dim = 784
        latent_dim = 2
        input_dims = (1, 28, 28)
    elif dataset == "Zilionis":
        input_dim = 306
        latent_dim = 2
        input_dims = (1, 306)
        num_labels = 20
    elif dataset == "PBMC":
        input_dim = 50
        latent_dim = 2
        input_dims = (1, 50)
        num_labels = 11
    elif dataset == "CElegans":
        input_dim = 100
        latent_dim = 2
        input_dims = (1, 100)
        num_labels = 37

    # Prepare SummaryWriter
    # writer = get_summary_writer(subdir=writer_dir)
    writer = None

    train_loader, test_loader = load_data(train_batch_size=train_batch_size,
                                          test_batch_size=256,
                                          dataset=dataset)

    # create model
    print(f"[model] move to {device}...")
    model = AE(input_shape=input_dim, latent_dim=latent_dim, input_dims=input_dims).to(device)  # , input_dims=(3, )
    model.load(model_path)

    # set and create path for saving model
    model_path_save = os.path.join(output_path,
                                   f"models/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/{writer_dir}")
    Path(model_path_save).mkdir(parents=True, exist_ok=True)

    # set and create path for saving images

    image_save_path = os.path.join(output_path,
                                   f"graphics/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/",
                                   img_path)
    Path(image_save_path).mkdir(parents=True, exist_ok=True)

    """
    Analyse Model
    """
    print("[model] analyze ...")

    # create colormap
    if dataset in ["PBMC", "Zilionis", "CElegans"]:
        seed = 0
        np.random.seed(seed)
        colors = get_distinct_colors(num_labels)
        np.random.shuffle(colors)
        cmap = ListedColormap(colors)
    else:
        cmap = "tab10"

    # number of steps for indicatrices
    if dataset == "Earth":
        num_steps = 5
    else:
        if model_name == "ParametricUMAP":
            num_steps = 20
        else:
            num_steps = 10

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
        if "determinants" in used_diagnostics:
            # determinants
            plot_determinants(model,
                              train_loader,
                              batch_size=500,  # TODO: wofür die batch size?
                              device=device,
                              quantile=.95,
                              scaling="log",
                              output_path_1=os.path.join(image_save_path, "det.png"),
                              output_path_2=os.path.join(image_save_path, "det_hist.png"),
                              writer=writer,
                              model_name=model_name,
                              dataset_name=dataset)

        if "indicatrices" in used_diagnostics:
            # calculate indicatrices
            plot_indicatrices(model,
                              train_loader,
                              device=device,
                              cmap=cmap,
                              num_steps=num_steps,
                              num_gon=num_gon,
                              model_name=model_name,
                              dataset_name=dataset,
                              output_path=os.path.join(image_save_path, "indicatrices.png"),
                              writer=writer)

        if "embedding" in used_diagnostics:
            # plot latent space
            plot_latent_space(model,
                              train_loader,
                              cmap=cmap,
                              dataset=dataset,
                              output_path=os.path.join(image_save_path, "latents.png"),
                              writer=writer)

    else:
        latent_data = genfromtxt(os.path.join("/".join(model_path.split("/")[:-1]), 'train_latents.csv'),
                                 skip_header=1,
                                 delimiter=',')

        latent_activations = latent_data[:, [0, 1]]
        labels = latent_data[:, 2]

        latent_activations = torch.from_numpy(latent_activations)
        labels = torch.from_numpy(labels)

        inputs, _, _, _ = data_forward(model, train_loader)

        if model_name == "PCA":
            if "indicatrices" in used_diagnostics:
                # calculate indicatrices
                plot_indicatrices(model,
                                  train_loader,
                                  device=device,
                                  cmap=cmap,
                                  num_steps=num_steps,
                                  num_gon=num_gon,

                                  dataset_name=dataset,
                                  output_path=os.path.join(image_save_path, "indicatrices.png"),
                                  writer=writer,
                                  model_name=model_name,
                                  latent_activations=latent_activations,
                                  labels=labels,
                                  inputs=inputs.cpu()
                                  )

            if "determinants" in used_diagnostics:
                # determinants
                plot_determinants(model,
                                  train_loader,
                                  batch_size=500,
                                  device=device,
                                  quantile=.97,
                                  scaling="log",
                                  output_path_1=os.path.join(image_save_path, "det.png"),
                                  output_path_2=os.path.join(image_save_path, "det_hist.png"),
                                  writer=writer,
                                  model_name=model_name,
                                  dataset_name=dataset,
                                  latent_activations=latent_activations)

        if "embedding" in used_diagnostics:
            # plot latent space
            plot_latent_space(model,
                              train_loader,
                              cmap=cmap,
                              dataset=dataset,
                              output_path=os.path.join(image_save_path, "latents.png"),
                              writer=writer,
                              latent_activations=latent_activations,
                              labels=labels)
