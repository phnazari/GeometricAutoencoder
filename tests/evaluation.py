import os
from datetime import datetime
from pathlib import Path
from decimal import Decimal

import conf
import numpy as np
import torch

from AutoEncoderVisualization.diagnostics.metric_properties import plot_determinants, \
    plot_indicatrices
from AutoEncoderVisualization.diagnostics.representations import plot_reconstruction, plot_latent_space, plot_dataset

from data.handle_data import load_data, data_forward
from firelight.visualizers.colorization import get_distinct_colors
from lib.TopoAE.src.models.submodules import DeepAE  # , ELUUMAPAutoEncoder
from matplotlib.colors import ListedColormap
from models import DeepThinAutoEncoder, SoftplusAE, ELUAutoEncoder, DeepThinSigmoidAutoEncoder, TestELUAutoEncoder, \
    ThinAutoEncoder, WideShallowAutoEncoder, WideDeepAutoEncoder, TopoAutoEncoder, \
    UMAPAutoEncoder, ELUUMAPAutoEncoder
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


def evaluate(alpha=None,
             beta=None,
             delta=None,
             writer_dir=None,
             epsilon=None,
             gamma=None,
             train=True,
             mode="normal",
             save=True,
             init=False,
             model_path=None,
             img_path=None,
             dataset="MNIST",
             model_name="Vanilla",
             used_diagnostics=None
             ):
    """
    Determine settings
    """

    if dataset == "Zilionis_normalized":
        dataset = "Zilionis"
    if dataset == "artificial":
        dataset = "Earth"

    conf.init()

    if model_name == "ParametricUMAP":
        from models import ELUUMAPAutoEncoder
        conf.topomodel = False
    else:
        from lib.TopoAE.src.models.submodules import ELUUMAPAutoEncoder
        conf.topomodel = True

    AE = ELUUMAPAutoEncoder

    # AE = ELUUMAPAutoEncoder

    # evaluate
    eval_model = True

    # dataset = "Earth"

    """ end modify """

    # set input dimensions
    if dataset in ["SwissRoll", "Mammoth", "HyperbolicParabloid", "Earth", "Spheres", "Images"]:
        input_dim = 3
        latent_dim = 2
        input_dims = (1, 3)

        train_batch_size = 125
    elif dataset in ["Saddle"]:
        input_dim = 3
        latent_dim = 2

        train_batch_size = 128
    elif dataset in ["chiocciola", "FigureEight"]:
        input_dim = 2
        latent_dim = 1

        train_batch_size = 512
    elif dataset in ["MNIST", "FashionMNIST"]:
        input_dim = 784
        latent_dim = 2
        input_dims = (1, 28, 28)

        train_batch_size = 256
    elif dataset in ["Zilionis"]:
        input_dim = 306
        latent_dim = 2
        input_dims = (1, 306)
        num_labels = 20

        train_batch_size = 256
    elif dataset in ["CIFAR10"]:
        input_dim = 3 * 32 * 32
        latent_dim = 2
        input_dims = (1, 306)

        train_batch_size = 256

    elif dataset in ["PBMC"]:
        input_dim = 50
        latent_dim = 2
        input_dims = (1, 50)
        num_labels = 11

        train_batch_size = 256

    elif dataset in ["CElegans"]:
        input_dim = 100
        latent_dim = 2
        input_dims = (1, 100)
        num_labels = 37

        train_batch_size = 125

    # path to save(d) model weights
    model_init_path = os.path.join(output_path, f"models/{dataset}/{AE.__name__}/init.pth")

    # if no model path to load from is passed and model should not be trained, then there is nothing to do
    if model_path is None and train is False:
        print("[exit] neither loading nor training new model")
        return

    if alpha == 0. and beta == 0. and delta == 0. and gamma == 0. and epsilon == 0. and mode != "vanilla":
        mode = "baseline"

    # Prepare SummaryWriter
    writer = get_summary_writer(subdir=writer_dir)
    print(f"[Writer] subdir {writer_dir}")

    """
    Import Data
    """

    train_loader, test_loader = load_data(train_batch_size=train_batch_size,
                                          test_batch_size=256,
                                          dataset=dataset)

    """
    Initialize Model
    """

    # create model
    print(f"[model] move to {device}...")
    model = AE(input_shape=input_dim, latent_dim=latent_dim, input_dims=input_dims).to(device)  # , input_dims=(3, )

    if model_path is not None:
        print("[model] load from path...")
        try:
            model.load(model_path)
            model_exists = True
        except FileNotFoundError:
            model_exists = False
    else:
        # if in train mode
        if os.path.isfile(model_init_path) and not init:
            print("[model] load existing init...")
            model.load(model_init_path)
        else:
            if not init:
                print("[model] saving new init...")
                torch.save(model.state_dict(), model_init_path)

        model_exists = True

    if save is True:
        # set and create path for saving model
        model_path_save = os.path.join(output_path,
                                       f"models/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/{writer_dir}")
        Path(model_path_save).mkdir(parents=True, exist_ok=True)
        model_path_save = os.path.join(model_path_save,
                                       f"{Decimal(alpha):.4e}_{Decimal(beta):.4e}_{Decimal(delta):.4e}_{Decimal(gamma):.4e}.pth")
    else:
        model_path_save = None

    # set and create path for saving images

    if img_path:
        image_save_path = os.path.join(output_path,
                                       f"graphics/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/",
                                       img_path)
    else:
        image_save_path = os.path.join(output_path,
                                       f"graphics/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/"
                                       f"{writer_dir}/{Decimal(alpha):.4e}_{Decimal(beta):.4e}_{Decimal(delta):.4e}_{Decimal(gamma):.4e}")

    Path(image_save_path).mkdir(parents=True, exist_ok=True)

    """
    Analyse Model
    """
    print("[model] analyze ...")

    if dataset in ["PBMC", "Zilionis", "CElegans"]:
        seed = 0
        np.random.seed(seed)
        colors = get_distinct_colors(num_labels)
        np.random.shuffle(colors)
        cmap = ListedColormap(colors)
    else:
        cmap = "tab10"

    if dataset == "Earth":
        num_steps = 5
    else:
        if model_name == "ParametricUMAP":
            num_steps = 20
        else:
            num_steps = 10

    num_gon = 500

    if eval_model:
        # preimage_of_ball(test_loader, model)
        # plot_pd(test_loader, model, 100)

        # circular_variance(test_loader,
        #                  model,
        #                  writer=writer,
        #                  output_path=os.path.join(image_save_path, "circular.png"))

        if input_dim == 3:
            # plot input dataset
            plot_dataset(model,
                         train_loader,
                         input_dim=input_dim,
                         writer=writer,
                         output_path=os.path.join(image_save_path, "dataset.png")
                         )

        if model_exists:
            if "determinants" in used_diagnostics:
                # determinants
                plot_determinants(model,
                                  train_loader,
                                  batch_size=500,
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
                plot_latent_space(model, train_loader, cmap=cmap, dataset=dataset,
                                  output_path=os.path.join(image_save_path, "latents.png"), writer=writer)

        else:
            latent_data = genfromtxt(os.path.join("/".join(model_path.split("/")[:-1]), 'train_latents.csv'),
                                     skip_header=1,
                                     delimiter=',')

            latent_activations = latent_data[:, [0, 1]]
            labels = latent_data[:, 2]

            latent_activations = torch.from_numpy(latent_activations)
            labels = torch.from_numpy(labels)

            inputs, _, _, _ = data_forward(model, train_loader)

            if "embedding" in used_diagnostics:
                # plot latent space
                plot_latent_space(model, train_loader, cmap=cmap, dataset=dataset,
                                  output_path=os.path.join(image_save_path, "latents.png"), writer=writer,
                                  latent_activations=latent_activations, labels=labels)

            if "indicatrices" in used_diagnostics and model_name == "PCA":
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

            if "determinants" in used_diagnostics and model_name == "PCA":
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
