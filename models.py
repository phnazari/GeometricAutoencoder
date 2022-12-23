import torch
from torch import nn

"""
Models
"""


class AutoEncoder(nn.Module):
    """
    Parent function for autoencoders which more sophisiticated models inherit from
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.input_dim = kwargs["input_shape"]
        self.latent_dim = kwargs["latent_dim"]

        self.latent_activations = None

    def register_hook(self):
        self.encoder.register_forward_hook(self.get_activation())

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def get_activation(self):
        """
        :return: activations at layer model
        """

        def hook(model, input, output):
            self.latent_activations = output
            self.latent_activations_detached = output.detach()

        return hook

    def forward(self, features):
        features = self.encoder(features)
        features = self.decoder(features)
        return features


class DeepThinAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=4),
            nn.ELU(),
            nn.Linear(in_features=4, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ELU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ELU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=4),
            nn.ELU(),
            nn.Linear(in_features=4, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.latent_dim, out_features=4),
            nn.ELU(),
            nn.Linear(in_features=4, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ELU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ELU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=4),
            nn.ELU(),
            nn.Linear(in_features=4, out_features=self.input_dim),
        )

        self.register_hook()


class ThinAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=4),
            nn.ELU(),
            nn.Linear(in_features=4, out_features=3),
            nn.ELU(),
            nn.Linear(in_features=3, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.latent_dim, out_features=3),
            nn.ELU(),
            nn.Linear(in_features=3, out_features=4),
            nn.ELU(),
            nn.Linear(in_features=4, out_features=self.input_dim),
        )

        self.register_hook()


class WideShallowAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=500),
            nn.ELU(),
            nn.Linear(in_features=500, out_features=250),
            nn.ELU(),
            nn.Linear(in_features=250, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.latent_dim, out_features=250),
            nn.ELU(),
            nn.Linear(in_features=250, out_features=500),
            nn.ELU(),
            nn.Linear(in_features=500, out_features=self.input_dim),
        )

        self.register_hook()


class TopoAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=1000),
            nn.ELU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.ELU(),
            nn.Linear(in_features=500, out_features=250),
            nn.ELU(),
            nn.Linear(in_features=250, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.latent_dim, out_features=250),
            nn.ELU(),
            nn.Linear(in_features=250, out_features=500),
            nn.ELU(),
            nn.Linear(in_features=500, out_features=1000),
            nn.ELU(),
            nn.Linear(in_features=1000, out_features=self.input_dim),
        )

        self.register_hook()


class ELUUMAPAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            # nn.Identity(),
            nn.Linear(self.latent_dim, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=self.input_dim),
        )

        self.register_hook()


class UMAPAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=self.input_dim),
        )

        self.register_hook()


class WideDeepAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=750),
            nn.ELU(),
            nn.Linear(in_features=750, out_features=500),
            nn.ELU(),
            nn.Linear(in_features=500, out_features=250),
            nn.ELU(),
            nn.Linear(in_features=250, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.latent_dim, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=250),
            nn.ELU(),
            nn.Linear(in_features=250, out_features=500),
            nn.ELU(),
            nn.Linear(in_features=500, out_features=750),
            nn.ELU(),
            nn.Linear(in_features=750, out_features=self.input_dim),
        )

        self.register_hook()


class DeepThinSigmoidAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        self.input_dim = kwargs["input_shape"]

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.input_dim, out_features=4),
            nn.Sigmoid(),
            nn.Linear(in_features=4, out_features=16),
            nn.Sigmoid(),
            nn.Linear(in_features=16, out_features=32),
            nn.Sigmoid(),
            nn.Linear(in_features=32, out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=32),
            nn.Sigmoid(),
            nn.Linear(in_features=32, out_features=16),
            nn.Sigmoid(),
            nn.Linear(in_features=16, out_features=4),
            nn.Sigmoid(),
            nn.Linear(in_features=4, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.latent_dim, out_features=4),
            nn.Sigmoid(),
            nn.Linear(in_features=4, out_features=16),
            nn.Sigmoid(),
            nn.Linear(in_features=16, out_features=32),
            nn.Sigmoid(),
            nn.Linear(in_features=32, out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=32),
            nn.Sigmoid(),
            nn.Linear(in_features=32, out_features=16),
            nn.Sigmoid(),
            nn.Linear(in_features=16, out_features=4),
            nn.Sigmoid(),
            nn.Linear(in_features=4, out_features=self.input_dim),
        )

        self.register_hook()


class LinearAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = kwargs["input_shape"]

        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=kwargs["input_shape"])
        )

        self.register_hook()


class SoftplusAE(AutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = kwargs["input_shape"]

        self.encoder = nn.Sequential(
            nn.Softplus(),
            nn.Linear(in_features=kwargs["input_shape"], out_features=128),
            nn.Softplus(),
            nn.Linear(in_features=128, out_features=32),
            nn.Softplus(),
            nn.Linear(in_features=32, out_features=8),
            nn.Softplus(),
            nn.Linear(in_features=8, out_features=self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Softplus(),
            nn.Linear(in_features=self.latent_dim, out_features=8),
            nn.Softplus(),
            nn.Linear(in_features=8, out_features=32),
            nn.Softplus(),
            nn.Linear(in_features=32, out_features=128),
            nn.Softplus(),
            nn.Linear(in_features=128, out_features=kwargs["input_shape"])
        )

        self.register_hook()


class ELUAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = kwargs["input_shape"]

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=kwargs["input_shape"], out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ELU(),
            nn.Linear(in_features=32, out_features=8),
            nn.ELU(),
            nn.Linear(in_features=8, out_features=self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=self.latent_dim, out_features=8),
            nn.ELU(),
            nn.Linear(in_features=8, out_features=32),
            nn.ELU(),
            nn.Linear(in_features=32, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=kwargs["input_shape"])
        )

        self.register_hook()


class TestELUAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = kwargs["input_shape"]

        self.encoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=kwargs["input_shape"], out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ELU(),
            nn.Linear(in_features=8, out_features=self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=self.latent_dim, out_features=8),
            nn.ELU(),
            nn.Linear(in_features=8, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=kwargs["input_shape"])
        )

        self.register_hook()


"""
Helpers
"""


def get_latent_activations(test_loader, device, model):
    """
    stacks the activations of latent space over test set
    :param test_loader: dataloader for test data
    :param device: device where computations should take place
    :param model: network model
    :return:
    """
    activations = None
    labels = None

    for k, (batch_features, batch_labels) in enumerate(test_loader):
        # do that in order to get activations at latent layer
        batch_features = batch_features.view(-1, model.input_dim).to(device)
        model(batch_features)

        if k == 0:
            activations = model.latent_activations
            labels = batch_labels
        else:
            activations = torch.vstack((activations, model.latent_activations))
            labels = torch.hstack((labels, batch_labels))

    return activations, labels
