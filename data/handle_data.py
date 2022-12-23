import conf
import torch
import torchvision

from conf import device
# from data.custom import SwissRoll, Spheres, Chiocciola, Saddle, HyperbolicParabloid, Earth, FigureEight, Zilionis, PBMC, \
#    CElegans, Images

from src.datasets.mnist import MNIST
from src.datasets.fashion_mnist import FashionMNIST
from src.datasets.celegans import CElegans
from src.datasets.zilionis import Zilionis
from src.datasets.pbmc import PBMC

from data.custom import WoollyMammoth, Earth


def load_data(train_batch_size=128, test_batch_size=32, dataset="MNIST", test_separate=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    test_separate = True

    if dataset == "MNIST":
        train_dataset = MNIST(train=True)
        test_dataset = MNIST(train=False)
    elif dataset == "FashionMNIST":
        train_dataset = FashionMNIST(train=True)
        test_dataset = FashionMNIST(train=False)
    elif dataset == "Zilionis":
        train_dataset = Zilionis(train=True)
        test_dataset = Zilionis(train=False)
    elif dataset == "PBMC":
        train_dataset = PBMC(train=True)
        test_dataset = PBMC(train=False)
    elif dataset == "CElegans":
        train_dataset = CElegans(train=True)
        test_dataset = CElegans(train=False)
    elif dataset == "Earth":
        train_dataset = Earth(train=True,
                              filename="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/landmass.pt")
        test_dataset = Earth(train=False,
                             filename="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/landmass.pt")
    else:
        return

    # While visualizing, we want to evaluate on the same data we trained on

    if not test_separate:
        test_dataset = train_dataset

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                               num_workers=16)  # shuffle=True, pin_memory=True

    #  for i, (batch_features, _) in enumerate(train_loader):
    #     # print(torch.max(batch_features), torch.min(batch_features))
    #     test = torch.mean(batch_features, dim=3)
    #     print(batch_features.shape, test.shape, torch.unique(test))

    return train_loader, test_loader


# TODO: move into base class of autoencoders
def data_forward(model, test_loader):
    inputs = torch.tensor([])
    outputs = torch.tensor([])
    latent_activations = torch.tensor([])
    labels = torch.tensor([])

    for k, (batch_features, batch_labels) in enumerate(test_loader):
        # do that in order to get activations at latent layer
        # print(batch_features.shape)

        if model.__class__.__name__ in ["DeepAE", "ELUUMAPAutoEncoder"] and conf.topomodel:  # and False:  # and False
            batch_features = batch_features.to(device)
            output = model.forward_(batch_features)
        else:
            batch_features = batch_features.view(-1, model.input_dim).to(device)
            output = model(batch_features)

        if k == 0:
            inputs = batch_features
            outputs = output
            latent_activations = model.latent_activations
            labels = batch_labels
        else:
            inputs = torch.vstack((inputs, batch_features))
            outputs = torch.vstack((outputs, output))
            latent_activations = torch.vstack((latent_activations, model.latent_activations))

            # vstack: batch_labels.shape = (256, 3)
            if batch_labels.ndim == 1:
                labels = torch.hstack((labels, batch_labels))
            else:
                labels = torch.vstack((labels, batch_labels))

    if model.__class__.__name__ in ["DeepAE", "ELUUMAPAutoEncoder"] and conf.topomodel:  # and False:
        outputs = outputs.view(-1, model.input_dim)

    return inputs, outputs, latent_activations, labels
