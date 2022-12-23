import os
from datetime import timedelta

from umap.parametric_umap import load_ParametricUMAP
import json

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch
from matplotlib import pyplot as plt
import dateutil.parser
from util import get_saving_kwargs

encoder_name = "sqrt"

input_dim = 784


def convert():
    paths = []

    for subdir, dirs, _ in os.walk(
            "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/fit_competitor/evaluation/repetitions/"):

        if len(subdir.split("/")) == 15 and subdir.split("/")[-1] == "ParametricUMAP":
            paths.append(subdir)

    from AutoEncoderVisualization.models import ELUUMAPAutoEncoder

    for i, path in enumerate(paths):
        print(f"{i} of {len(paths)}")

        model = path.split("/")[-2]
        print(model)

        if model in ["MNIST", "FashionMNIST"]:
            dimension = 784
        elif model == "Zilionis_normalized":
            dimension = 306
        elif model == "PBMC":
            dimension = 50
        elif model == "CElegans":
            dimension = 100
        elif model == "Earth":
            continue

        embedder = load_ParametricUMAP(os.path.join(path, "model"))
        model = ELUUMAPAutoEncoder(input_shape=dimension, latent_dim=2)

        model.encoder[1].weight = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[2].weights[0].numpy()).T)
        model.encoder[3].weight = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[3].weights[0].numpy()).T)
        model.encoder[5].weight = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[4].weights[0].numpy()).T)
        model.encoder[7].weight = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[5].weights[0].numpy()).T)
        model.encoder[9].weight = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[6].weights[0].numpy()).T)
        model.encoder[1].bias = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[2].bias.numpy()))
        model.encoder[3].bias = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[3].bias.numpy()))
        model.encoder[5].bias = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[4].bias.numpy()))
        model.encoder[7].bias = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[5].bias.numpy()))
        model.encoder[9].bias = torch.nn.Parameter(torch.from_numpy(embedder.encoder.layers[6].bias.numpy()))

        model.decoder[1].weight = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[1].weights[0].numpy()).T)
        model.decoder[3].weight = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[2].weights[0].numpy()).T)
        model.decoder[5].weight = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[3].weights[0].numpy()).T)
        model.decoder[7].weight = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[4].weights[0].numpy()).T)
        model.decoder[9].weight = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[5].weights[0].numpy()).T)
        model.decoder[1].bias = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[1].bias.numpy()))
        model.decoder[3].bias = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[2].bias.numpy()))
        model.decoder[5].bias = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[3].bias.numpy()))
        model.decoder[7].bias = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[4].bias.numpy()))
        model.decoder[9].bias = torch.nn.Parameter(torch.from_numpy(embedder.decoder.layers[5].bias.numpy()))

        torch.save(model.state_dict(), os.path.join(path, "model_state.pth"))


def plot_loss_curves():
    paths = [
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep1/MNIST/Vanilla/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep2/MNIST/Vanilla/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep3/MNIST/Vanilla/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep4/MNIST/Vanilla/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep5/MNIST/Vanilla/geometric_loss.pth",

        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep1/MNIST/GeomReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep2/MNIST/GeomReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep3/MNIST/GeomReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep4/MNIST/GeomReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep5/MNIST/GeomReg/geometric_loss.pth",

        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep1/MNIST/TopoReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep2/MNIST/TopoReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep3/MNIST/TopoReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep4/MNIST/TopoReg/geometric_loss.pth",
        "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/geometric_loss/repetitions/rep5/MNIST/TopoReg/geometric_loss.pth",
    ]

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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("epoch")
    ax.set_ylabel("Geometric Loss")

    for model, loss in losses.items():
        if loss:
            loss = torch.stack(loss)
            mean_loss = torch.mean(loss, dim=0)
            std_loss = torch.std(loss, dim=0)
            x = torch.arange(len(mean_loss))

            ax.plot(x, mean_loss, label=key_to_legend[model], color=key_to_color[model])
            ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, alpha=0.5, color=key_to_color[model])
            ax.set_yscale("log")

    plt.legend(loc="best")

    output_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/graphics/MNIST/Loss/geometric_loss.png"
    plt.savefig(output_path, **get_saving_kwargs())

    plt.show()


def calculate_runtime():
    paths = []

    for subdir, dirs, _ in os.walk(
            "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/speed/repetitions/"):
        if len(subdir.split("/")) == 15:
            paths.append(os.path.join(subdir, "run.json"))

    for subdir, dirs, _ in os.walk(
            "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/fit_competitor/speed/repetitions/"):
        if len(subdir.split("/")) == 15:
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

        with open(path, "r") as file:
            data = json.loads(file.read())

            start_time_str = data["start_time"].split("T")[-1]
            end_time_str = data["stop_time"].split("T")[-1]

            start_time = dateutil.parser.parse(start_time_str)
            end_time = dateutil.parser.parse(end_time_str)

            time_delta = end_time - start_time

            if time_delta.days == -1:
                time_delta += timedelta(days=1)

            time_delta_minutes = time_delta.total_seconds() / 60

            times[model].append(time_delta_minutes)

    for model, minutes in times.items():
        minutes = torch.tensor(minutes)
        mean_minutes = torch.mean(minutes, dim=0)
        std_minutes = torch.std(minutes, dim=0)

        print(model, mean_minutes, std_minutes)
