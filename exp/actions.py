import os
import pandas as pd
import matplotlib as mpl
from src.models.submodules import BoxAutoEncoder, ConvolutionalAutoEncoder
from util import get_saving_kwargs, get_sc_kwargs, get_coordinates
import dateutil.parser
from matplotlib import pyplot as plt
import torch
from datetime import timedelta

from firelight.visualizers.colorization import get_distinct_colors
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
import numpy as np
from conf import device
from src.datasets.pbmc_new import PBMC_new
from data.handle_data import load_data, data_forward
from src.diffgeo.connections import LeviCivitaConnection
from src.diffgeo.manifolds import RiemannianManifold
from src.diffgeo.metrics import PullbackMetric
from src.models import GeometricAutoencoder
from umap.parametric_umap import load_ParametricUMAP, ParametricUMAP
import json

os.environ["GEOMSTATS_BACKEND"] = "pytorch"


def convert():
    paths = []

    for subdir, dirs, _ in os.walk(
            os.path.join(os.path.dirname(__file__), '..', 'experiments/fit_competitor/evaluation/repetitions/')):

        if len(subdir.split("/")) == 15 and subdir.split("/")[-1] == "ParametricUMAP":
            paths.append(subdir)

    for i, path in enumerate(paths):
        print(f"{i} of {len(paths)}")

        model = path.split("/")[-2]

        if model in ["MNIST", "FashionMNIST"]:
            dimension = 784
            input_dims = (1, 28, 28)
        elif model == "Zilionis_normalized":
            dimension = 306
            input_dims = (1, 306)
        elif model == "PBMC":
            dimension = 50
            input_dims = (1, 50)
        elif model == "PBMC_new":
            dimension = 50
            input_dims = (1, 50)
        elif model == "CElegans":
            dimension = 100
            input_dims = (1, 100)
        elif model == "Earth":
            continue
        else:
            dimension = 784
            input_dims = (1, 28, 28)

        # if model != "PBMC_new":
        #    continue

        embedder = load_ParametricUMAP(os.path.join(path, "model"))

        # model = BoxAutoEncoder(input_shape=dimension, latent_dim=2)
        model = BoxAutoEncoder(input_dims=input_dims,
                               input_shape=dimension, latent_dim=2)

        model.encoder[1].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[2].weights[0].numpy()).T)
        model.encoder[3].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[3].weights[0].numpy()).T)
        model.encoder[5].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[4].weights[0].numpy()).T)
        model.encoder[7].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[5].weights[0].numpy()).T)
        model.encoder[9].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[6].weights[0].numpy()).T)
        model.encoder[1].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[2].bias.numpy()))
        model.encoder[3].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[3].bias.numpy()))
        model.encoder[5].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[4].bias.numpy()))
        model.encoder[7].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[5].bias.numpy()))
        model.encoder[9].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.encoder.layers[6].bias.numpy()))

        model.decoder[1].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[1].weights[0].numpy()).T)
        model.decoder[3].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[2].weights[0].numpy()).T)
        model.decoder[5].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[3].weights[0].numpy()).T)
        model.decoder[7].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[4].weights[0].numpy()).T)
        model.decoder[9].weight = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[5].weights[0].numpy()).T)
        model.decoder[1].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[1].bias.numpy()))
        model.decoder[3].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[2].bias.numpy()))
        model.decoder[5].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[3].bias.numpy()))
        model.decoder[7].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[4].bias.numpy()))
        model.decoder[9].bias = torch.nn.Parameter(
            torch.from_numpy(embedder.decoder.layers[5].bias.numpy()))

        torch.save(model.state_dict(), os.path.join(path, "model_state.pth"))


def plot_loss_curves():
    models = ["Vanilla", "TopoReg", "GeomReg"]
    paths = []

    for model in models:
        for rep in range(1, 6):
            path = os.path.join(os.path.dirname(__file__), '..',
                                f'experiments/train_model/geometric_loss/repetitions/rep{rep}/MNIST/{model}/geometric_loss.pth')
            paths.append(path)

    losses = {
        "Vanilla": [],
        "GeomReg": [],
        "TopoReg": [],
    }

    key_to_legend = {
        "Vanilla": "Vanilla",
        "GeomReg": "Geometric",
        "TopoReg": "Topological"
    }

    key_to_color = {
        "Vanilla": "navy",
        "GeomReg": "seagreen",
        "TopoReg": "maroon"
    }

    for path in paths:
        model = path.split("/")[-2]
        geom_loss = torch.load(path)
        losses[model].append(geom_loss)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fontdict = {'family': 'DeJavu Serif',
                # 'serif': 'Times New Roman',
                'size': 15}

    ax.set_xlabel("Epoch", fontdict=fontdict)
    ax.set_ylabel("Geometric Loss", fontdict=fontdict)
    ax.tick_params(axis='both', which='major', labelsize=15)

    for model, loss in losses.items():
        if loss:
            loss = torch.stack(loss)
            mean_loss = torch.mean(loss, dim=0)
            std_loss = torch.std(loss, dim=0)
            x = torch.arange(len(mean_loss))

            ax.plot(x, mean_loss,
                    label=key_to_legend[model], color=key_to_color[model])
            ax.fill_between(x, mean_loss - std_loss, mean_loss +
                            std_loss, alpha=0.5, color=key_to_color[model])
            ax.set_yscale("log")

    mean_pumap_loss, std_pumap_loss = calc_pumap_geom_loss()
    # mean_pumap_loss, std_pumap_loss = [100], [2]

    ax.errorbar(150, mean_pumap_loss, yerr=std_pumap_loss, label="UMAP AE", color="darkgoldenrod", capsize=10,
                elinewidth=2, ms=10, marker=".")

    x_int = [0, 20, 40, 60, 80, 100, 150]
    x_str = [0, 20, 40, 60, 80, 100, "UMAP AE"]

    ax.set_xticks(x_int, fontdict=fontdict)
    ax.set_xticklabels(x_str, fontdict=fontdict)

    chartBox = ax.get_position()
    # ax.set_position([0, 0, chartBox.width, chartBox.height])

    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22, markerscale=2)

    plt.legend(loc="best")

    ax.spines[['right', 'top']].set_visible(False)

    output_path = "exp/output/graphics/MNIST/Loss/geometric_loss.png"
    plt.savefig(output_path, **get_saving_kwargs())

    plt.show()


def calculate_runtime():
    paths = []

    for subdir, dirs, _ in os.walk(
            os.path.join(os.path.dirname(__file__), '..', "experiments/train_model/speed/repetitions/")):
        if len(subdir.split("/")) == 15 and subdir.split("/")[-1] != "train_model":
            paths.append(os.path.join(subdir, "run.json"))

    for subdir, dirs, _ in os.walk(
            os.path.join(os.path.dirname(__file__), '..', "experiments/fit_competitor/speed/repetitions/")):
        if len(subdir.split("/")) == 15 and subdir.split("/")[-1] != "fit_competitor":
            paths.append(os.path.join(subdir, "run.json"))

    times = {
        "Vanilla": [],
        "GeomReg": [],
        "TopoReg": [],
        "PCA": [],
        "TSNE": [],
        "ParametricUMAP": [],
        "UMAP": []
    }

    for path in paths:
        model = path.split("/")[-2]
        dataset = path.split("/")[-3]

        with open(path, "r") as file:
            data = json.loads(file.read())

            start_time_str = data["start_time"].split("T")[-1]
            end_time_str = data["stop_time"].split("T")[-1]

            start_time = dateutil.parser.parse(start_time_str)
            end_time = dateutil.parser.parse(end_time_str)

            time_delta = end_time - start_time

            if model == "UMAP":
                print(dataset, time_delta, end_time, start_time)

            if time_delta.days == -1:
                time_delta += timedelta(days=1)

            time_delta_minutes = time_delta.total_seconds() / 60

            times[model].append(time_delta_minutes)

    for model, minutes in times.items():
        minutes = torch.tensor(minutes)
        mean_minutes = torch.mean(minutes, dim=0)
        std_minutes = torch.std(minutes, dim=0)

        print(model, mean_minutes, std_minutes)


def calc_pumap_geom_loss():
    paths = [
        os.path.join(os.path.dirname(__file__), "..",
                     "experiments/fit_competitor/evaluation/repetitions/rep1/MNIST/ParametricUMAP/model_state.pth"),
        os.path.join(os.path.dirname(__file__), "..",
                     "experiments/fit_competitor/evaluation/repetitions/rep2/MNIST/ParametricUMAP/model_state.pth"),
        os.path.join(os.path.dirname(__file__), "..",
                     "experiments/fit_competitor/evaluation/repetitions/rep4/MNIST/ParametricUMAP/model_state.pth"),
        os.path.join(os.path.dirname(__file__), "..",
                     "experiments/fit_competitor/evaluation/repetitions/rep5/MNIST/ParametricUMAP/model_state.pth"),
    ]

    train_loader, _ = load_data(train_batch_size=125,
                                test_batch_size=125,
                                dataset="MNIST")

    geom_errors = []

    for i, path in enumerate(paths):
        print(path)
        model = GeometricAutoencoder(
            lam=0.1, autoencoder_model="BoxAutoEncoder")
        model.autoencoder.load(path)

        geom_error = 0

        for batch, (img, label) in enumerate(train_loader):
            model.train()
            loss, loss_components = model(img)
            geom_error += loss_components["loss.geom_error"]

        print(geom_error)

        geom_errors.append(geom_error)

    geom_errors = torch.tensor(geom_errors)

    mean_geom_error = torch.mean(geom_errors)
    std_geom_error = torch.std(geom_errors)

    return mean_geom_error, std_geom_error


def create_colorbar():
    root_path = os.path.join(os.path.dirname(__file__), 'output/graphics/cbar')
    sizes = [10, 15, 20, 25, 30, 35]
    # sizes = [10]

    for size in sizes:
        # Vertical
        path = os.path.join(root_path, f'cbar_vertical_{size}.png')

        # Make a figure and axes with dimensions as desired.
        # fig = plt.figure(figsize=(8, 1.5))
        fig = plt.figure(figsize=(2., 8))
        # ax1 = fig.add_axes([0.05, 0.75, 0.9, 0.15])
        ax1 = fig.add_axes([0.05, 0.05, 0.15, 0.9])
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        cmap_old = plt.get_cmap("turbo")

        minval = 0.4  # 0.2
        maxval = 0.8  # 0.6
        n = 100
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap_old.name, a=minval, b=maxval),
            cmap_old(np.linspace(minval, maxval, n)))

        norm = mpl.colors.Normalize(vmin=-1.8, vmax=1.22)

        cb1 = mpl.colorbar.ColorbarBase(ax1,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical',
                                        )

        cb1.ax.set_yticks([-1.0, 0., 1.1])

        cb1.ax.set_yticklabels(["contract", "neutral", "expand"], rotation=90)
        cb1.ax.tick_params(labelsize=size)
        # cb1.set_label('Scaled Generalized Jacobian Determinant', size=size)

        plt.savefig(path, format="png", pad_inches=0, dpi=200)

        plt.show()

        # Horizontal
        path = os.path.join(root_path, f'cbar_horizontal_{size}.png')

        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(8, 1.5))
        ax1 = fig.add_axes([0.05, 0.75, 0.9, 0.15])

        cmap = "turbo"
        norm = mpl.colors.Normalize(vmin=-1.8, vmax=1.22)

        cb1 = mpl.colorbar.ColorbarBase(ax1,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal',
                                        )

        cb1.ax.set_xticks([-1., 0., 1.])
        cb1.ax.set_xticklabels(
            ["-1\n(contract)", "0\n(neutral)", " 1\n(expand)"])
        cb1.ax.tick_params(labelsize=size)

        plt.savefig(path, format="png", pad_inches=0, dpi=200)

        plt.show()

        # Horizontal Cropped
        path = os.path.join(root_path, f'cbar_horizontal_cropped_{size}.png')

        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(8, 1.5))
        ax1 = fig.add_axes([0.05, 0.75, 0.9, 0.15])

        cmap_old = plt.get_cmap("turbo")

        minval = 0.3  # 0.2
        maxval = 0.8  # 0.6
        n = 100
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap_old.name, a=minval, b=maxval),
            cmap_old(np.linspace(minval, maxval, n)))

        norm = mpl.colors.Normalize(vmin=-1.8, vmax=1.22)

        cb1 = mpl.colorbar.ColorbarBase(ax1,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal',
                                        )

        cb1.ax.set_xticks([-1.3, 0., 0.8])
        cb1.ax.set_xticklabels(["contract", "neutral", " expand"])
        cb1.ax.tick_params(labelsize=size)

        plt.savefig(path, format="png", pad_inches=0, dpi=200)

        plt.show()


def pullback_metric_condition():
    # num_steps = 20
    coords0 = None
    dataset_name = "MNIST"

    input_dim = 784
    latent_dim = 2
    input_dims = (1, 28, 28)

    base = os.path.join(os.path.dirname(__file__), '..', "experiments")

    condition_numbers = {
        "Vanilla": [],
        "GeomReg": [],
        "TopoReg": [],
        "ParametricUMAP": []
    }

    subdir = "evaluation"

    model_paths = [
        os.path.join(base, "train_model", subdir, "repetitions/rep1",
                     dataset_name, "Vanilla", "model_state.pth"),
        os.path.join(base, "train_model", subdir, "repetitions/rep1",
                     dataset_name, "GeomReg", "model_state.pth"),
        os.path.join(base, "train_model", subdir, "repetitions/rep1",
                     dataset_name, "TopoReg", "model_state.pth"),
        os.path.join(base, "fit_competitor", subdir, "repetitions/rep1",
                     dataset_name, "ParametricUMAP", "model_state.pth"),
    ]

    for model_path in model_paths:
        model = ConvolutionalAutoEncoder(
            input_shape=input_dim, latent_dim=latent_dim, input_dims=input_dims).to(device)
        model.load(model_path)

        model_name = model_path.split("/")[-2]

        train_loader, _ = load_data(train_batch_size=256,
                                    test_batch_size=256,
                                    dataset=dataset_name)

        _, _, latent_activations, _ = data_forward(model, train_loader)
        latent_activations = latent_activations.detach().cpu()

        # if taken from a regular grid
        num_steps = 100
        coordinates = get_coordinates(torch.squeeze(latent_activations),
                                      grid="convex_hull",
                                      num_steps=num_steps,
                                      coords0=coords0,
                                      dataset_name=dataset_name,
                                      model_name=model_name).to(device)

        # if taken from the data
        # n_samples = len(latent_activations)
        # generator = torch.Generator().manual_seed(0)
        # perm = torch.randperm(latent_activations.shape[0], generator=generator)
        # coordinates = latent_activations[perm[:n_samples]].to(device)

        pbm = PullbackMetric(2, model.immersion)
        lcc = LeviCivitaConnection(2, pbm)
        rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)
        # metric at the point
        metric_matrices = rm.metric.metric_matrix(coordinates)

        cond = torch.linalg.cond(metric_matrices)

        condition_numbers[model_name] = cond

    mean_cond_van = torch.mean(condition_numbers["Vanilla"])
    std_cond_van = torch.std(condition_numbers["Vanilla"])
    mean_cond_geom = torch.mean(condition_numbers["GeomReg"])
    std_cond_geom = torch.std(condition_numbers["GeomReg"])
    mean_cond_topo = torch.mean(condition_numbers["TopoReg"])
    std_cond_topo = torch.std(condition_numbers["TopoReg"])
    mean_cond_umap = torch.mean(condition_numbers["ParametricUMAP"])
    std_cond_umap = torch.std(condition_numbers["ParametricUMAP"])

    print("Vanilla", mean_cond_van, std_cond_van)
    print("GeomReg", mean_cond_geom, std_cond_geom)
    print("TopoReg", mean_cond_topo, std_cond_topo)
    print("PUMAP", mean_cond_umap, std_cond_umap)


def hdb_scan():
    dirnames = "~/workspace/AutoEncoderVisualization/experiments/train_model/evaluation/repetitions/"

    def get_score(dirname):
        df = pd.read_csv(os.path.join(dirname, "test_latents.csv"))
        embeddings = torch.tensor(df[["0", "1"]].values)
        labels_true = torch.tensor(df["labels"].values)

        clusterer = hdbscan.HDBSCAN(approx_min_span_tree=False)

        clusterer.fit(embeddings)

        labels = clusterer.labels_

        # remove noise points
        if len(embeddings[labels != -1]) > 0:
            NN = NearestNeighbors(n_neighbors=1)
            NN.fit(embeddings[labels != -1])
            nbrs_idx = NN.kneighbors(
                embeddings[labels == -1], return_distance=False).reshape(-1)
            labels[labels == -1] = labels[labels != -1][nbrs_idx]

            score = adjusted_rand_score(labels_true, labels)
        else:
            score = np.nan

        return score

    datasets = ["MNIST", "FashionMNIST", "CElegans", "PBMC", "Zilionis"]
    models = ["Vanilla", "GeomReg"]
    for dataset in datasets:
        for model in models:
            scores = []
            for i in range(1, 6):
                dirname = os.path.join(dirnames, f"rep{i}", dataset, model)
                score = get_score(dirname)
                scores.append(score)
                # mean_errors, std_errors = analyze_scores(dataset, model)

            scores = np.array(scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            print(dataset, model, mean_score, std_score)
            
            # errors.append(np.array(mean_errors)[~np.isnan(mean_errors)].flatten())
            # errors.append(np.array(mean_errors).flatten())
            # print(dataset, model, mean_errors, std_errors)

def test_cluster_2():
    plot_imgs = True
    data = "test"
    model = "GeomReg"

    digit = 5

    dirname = f"~/workspace/AutoEncoderVisualization/experiments/train_model/evaluation/repetitions/rep1/MNIST/{model}"

    # read in data
    df = pd.read_csv(os.path.join(dirname, f"{data}_latents.csv"))
    embeddings = torch.tensor(df[["0", "1"]].values)
    labels_true = torch.tensor(df["labels"].values)

    if data == "test":
        df_inp = pd.read_csv(os.path.join(
            dirname, f"{data}_inputs.csv")).iloc[:, :-1]
        inputs = torch.tensor(df_inp.values)
        inputs = inputs.view(len(inputs), 28, 28)

    # filter out the 2s
    embeddings = embeddings[labels_true == digit]
    if data == "test":
        inputs = inputs[labels_true == digit]
    labels_true = labels_true[labels_true == digit]

    # differentiate clusters
    if model == "Vanilla":
        if digit == 2:
            right_cluster = embeddings[:, 0] > 8
            left_cluster = embeddings[:, 0] < 8
        elif digit == 5:
            right_cluster = embeddings[:, 1] > 0.2
            left_cluster = embeddings[:, 1] < 0.2
    elif model == "GeomReg":
        if digit == 2:
            left_cluster = embeddings[:, 1] < 2.3
            right_cluster = embeddings[:, 1] > 2.3
        else:
            right_cluster = embeddings[:, 0] > 3.3
            left_cluster = embeddings[:, 0] < 3.3
    elif model == "TopoReg":
        right_cluster = embeddings[:, 1] < 0.1
        left_cluster = embeddings[:, 1] > 0.1

    print(model, torch.sum(right_cluster)/torch.sum(left_cluster))

    if data == "test":
        right_input = inputs[right_cluster]
        left_input = inputs[left_cluster]

    if plot_imgs:
        seed = 2
        np.random.seed(seed)
        colors = get_distinct_colors(10)
        np.random.shuffle(colors)
        cmap = ListedColormap(colors)

        fig, ax = plt.subplots()
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c="fuchsia", s=0.1)
        ax.set_aspect('equal')
        plt.savefig(
            f"/export/home/pnazari/workspace/AutoEncoderVisualization/exp/output/{digit}cluster/cluster_{digit}_{model}.png", **get_saving_kwargs())

        if data == "test":
            fig = plt.figure(figsize=(8, 8))

            w = 10
            h = 10
            columns = 4
            rows = 5
            for i in range(1, columns*rows + 1):
                img = np.random.randint(10, size=(h, w))
                fig.add_subplot(rows, columns, i)
                plt.imshow(right_input[np.random.randint(len(right_input))], cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
            plt.savefig(
                f"/export/home/pnazari/workspace/AutoEncoderVisualization/exp/output/{digit}cluster/right_cluster_{model}.png", **get_saving_kwargs())

            fig = plt.figure(figsize=(8, 8))
            for i in range(1, columns*rows + 1):
                img = np.random.randint(10, size=(h, w))
                fig.add_subplot(rows, columns, i)
                plt.imshow(left_input[np.random.randint(len(left_input))], cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
            plt.savefig(
                f"/export/home/pnazari/workspace/AutoEncoderVisualization/exp/output/{digit}cluster/left_cluster_{model}.png", **get_saving_kwargs())


hdb_scan()
