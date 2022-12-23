import os
import torch

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from evaluation import evaluate

# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Images/ELUUMAPAutoEncoder/2022.10.27/test/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/SwissRoll/ELUUMAPAutoEncoder/2022.10.27/test/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/test_runs/2/model_runs/5/model_state.pth"


# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Earth/ELUUMAPAutoEncoder/2022.10.31/Earth/alpha/1.0000e-1_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/repetitions/rep1/MNIST/Vanilla/model_state.pth"
# model_path = "/export/ial-nfs/user/pnazari/results/train_model/repetitions/rep1/MNIST/GeomReg/model_state.pth"

model_paths = []

datasets = [
    "MNIST",
    # "Earth",
    "FashionMNIST",
    "CElegans",
    "Zilionis_normalized",
    "PBMC"
]

models = [
    #"Vanilla",
    #"TopoReg",
    #"GeomReg",
    #"PCA",
    #"TSNE",
    #"UMAP",
    "ParametricUMAP"
]

diagnostics = [
    "indicatrices",
    "determinants",
    "embedding"
]

base = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/"

for dataset in datasets:
    for model in models:
        if model in ["Vanilla", "TopoReg", "GeomReg"]:
            dir = "train_model"
        else:
            dir = "fit_competitor"

        model_paths.append(os.path.join(base, dir, "evaluation/repetitions/rep1", dataset, model, "model_state.pth"))

for i, model_path in enumerate(model_paths):
    img_path = model_path.split("/")[-2]
    dataset = model_path.split("/")[-3]
    model_name = model_path.split("/")[-2]
    # print(img_path)
    # img_path = img_paths[i]

    evaluate(alpha=0., beta=0., delta=0., epsilon=0., gamma=0., writer_dir="test",
             epochs=100,
             mode="normal",
             create_video=False,
             train=False,
             model_path=model_path,
             save=False,
             std=1,
             weight_decay=1e-5,
             n_gaussian_samples=10,
             n_origin_samples=64,
             img_path=img_path,
             model_name=model_name,
             dataset=dataset,
             used_diagnostics=diagnostics
             )

# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Earth/ELUUMAPAutoEncoder/2022.11.21/test/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/ial-nfs/user/pnazari/results/train_model/repetitions/rep1/MNIST/GeomReg/model_state.pth"

# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/fit_competitor/repetitions/rep1/PBMC/PCA/model_state.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/repetitions/rep1/PBMC/Vanilla/model_state.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/artificial/repetitions/rep1/Earth/GeomReg/model_state.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/artificial/repetitions/rep2/Earth/GeomReg/model_state.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/artificial/repetitions/rep3/Earth/GeomReg/model_state.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/test_runs/3/model_state.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/lib/TopoAE/experiments/train_model/artificial/repetitions/rep3/Earth/Vanilla/model_state.pth"
