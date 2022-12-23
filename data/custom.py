import json
import os
import pyreadr
from abc import ABCMeta, abstractmethod

from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, random_split

from util import minmax, cmap_labels


class CustomDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, n_samples=0, noise=0.0, train=True):
        """
        Create a SwissRoll Dataset
        :param n_samples: number of samples for one role
        :param noise: standard deviation of the gaussian noise
        """

        super().__init__()

        # if n_samples != 0:
        self.n_samples = n_samples
        self.noise = noise
        self.train = train
        self.train_dataset_size = 10000

        self.dataset, self.coordinates = self.create()

        if len(torch.unique(self.coordinates)) > 1:
            self.labels = self.minmax(self.coordinates)
        else:
            self.labels = self.coordinates

        self.labels = self.coordinates

        # truncate dataset
        if n_samples != 0:
            test_size = int(0.1 * n_samples)
            train_size = n_samples - test_size
        else:
            test_size = int(0.1 * len(self.dataset))
            train_size = len(self.dataset) - test_size

        rest = len(self.dataset) - test_size - train_size

        train_subset, test_subset, _ = torch.utils.data.random_split(self.dataset,
                                                                     [train_size, test_size, rest],
                                                                     generator=torch.Generator().manual_seed(42))

        # g = torch.Generator()
        # g.manual_seed(seed)
        # test_idx = torch.randperm(len(self.dataset), generator=g)[:test_size]

        if self.train:
            self.n_samples = len(self.dataset)
        else:
            self.n_samples = test_size
            self.dataset = self.dataset[test_subset.indices]
            self.labels = self.labels[test_subset.indices]

        # if self.n_samples != 0:
        #    test_dataset = self.dataset[:self.n_samples]
        #    test_labels = self.labels[:self.n_samples]

        # print(torch.any(self.dataset != test_dataset))
        # print(torch.any(self.labels != test_labels))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        item = self.dataset[index]
        label = self.labels[index]

        return item, label

    @staticmethod
    def transform_labels(labels):
        return cmap_labels(labels)

    @staticmethod
    def minmax(item):
        return minmax(item)

    @abstractmethod
    def create(self):
        """
        Create the dataset
        """


class SwissRoll(CustomDataset):
    """
    Create a SwissRoll Dataset
    """

    def create(self):
        """
        Generate a swiss roll dataset.
        """

        t = 1.5 * np.pi * (1 + 2 * torch.rand(self.n_samples))
        y = 21 * torch.rand(self.n_samples)

        x = t * np.cos(t)
        z = t * np.sin(t)

        X = torch.stack((x, y, z), dim=-2)

        X += self.noise * torch.normal(mean=torch.zeros((3, self.n_samples)),
                                       std=torch.ones((3, self.n_samples)))

        X = X.T

        t = torch.squeeze(t)

        return X, t


class HyperbolicParabloid(CustomDataset):
    """
    Create a hyperbolicparabloid dataset
    """

    def create(self):
        """
        Generate a hyperbolicparabloid roll dataset.
        """

        x = -2 * torch.rand(self.n_samples) + 1
        y = -2 * torch.rand(self.n_samples) + 1

        z = x ** 2 - y ** 2

        X = torch.stack((x, y, z), dim=-2)

        X += self.noise * torch.normal(mean=torch.zeros((3, self.n_samples)),
                                       std=torch.ones((3, self.n_samples)))

        X = X.T

        t = torch.squeeze(z)

        return X, t


class Saddle(CustomDataset):
    """
    Create a saddle dataset
    """

    def create(self):
        """
        Generate a swiss roll dataset.
        """

        x = -2 * torch.rand(self.n_samples) + 1
        y = -2 * torch.rand(self.n_samples) + 1

        z = x ** 2 + y ** 3

        X = torch.stack((x, y, z), dim=-2)

        X += self.noise * torch.normal(mean=torch.zeros((3, self.n_samples)),
                                       std=torch.ones((3, self.n_samples)))

        X = X.T

        t = torch.squeeze(z)

        return X, t


class Chiocciola(CustomDataset):
    """
    Create a snail dataset
    """

    def create(self):
        """
        Generate a swiss snail dataset.
        """

        t = 1.5 * np.pi * (1 + 2 * torch.rand(self.n_samples))

        x = t * np.cos(t)
        y = t * np.sin(t)

        X = torch.stack((x, y))

        # X += self.noise * torch.normal(mean=torch.zeros((self.n_rolls, 3, self.n_samples)),
        #                                std=torch.ones((self.n_rolls, 3, self.n_samples)))

        X = X.T

        t = torch.squeeze(t)

        return X, t


class WoollyMammoth(Dataset):
    def __init__(self):
        """
        Create a Woolly Mammoth Dataset
        """

        super().__init__()

        self.file = "/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/mammoth/mammoth_umap.json"

        self.dataset, self.labels = self.load()
        self.size = self.dataset.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = self.dataset[index]
        label = self.labels[index]
        return item, label

    @staticmethod
    def transform_labels(labels):
        COLORS = torch.tensor([
            [204, 0, 17],
            [0, 34, 68],
            [221, 221, 51],
            [0, 68, 136],
            [0, 51, 17],
            [17, 170, 170],
            [46, 102, 15],
            [155, 32, 6],
            [204, 0, 17],
            [17, 170, 170],
            [17, 170, 170]
        ])

        return COLORS[labels]

    def load(self):
        """
        Load mammoth dataset
        """

        f = open(self.file)
        raw_data = json.load(f)
        f.close()

        dataset = torch.tensor(raw_data["3d"])

        labels = torch.tensor(raw_data["labels"])

        return dataset, labels


class Spheres(Dataset):
    def __init__(self, n_samples=100, n_spheres=11, radius=5, noise=None):
        """
        Create a dataset 2-spheres embedded in three-dimensional space
        """

        super().__init__()

        self.n_samples = n_samples
        self.dim = 2
        self.n_spheres = n_spheres
        self.radius = radius
        self.noise = noise

        self.dataset, self.labels = self.create()

        self.size = self.dataset.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = self.dataset[index]
        label = self.labels[index]
        return item, label

    def create(self):
        """
        Create the dataset
        :return: dataset, labels
        """

        spheres = []
        labels = []
        for i in range(self.n_spheres):
            origin = torch.tensor([i * 3 * self.radius, 0, 0])
            sphere = self.dsphere(n=self.n_samples, d=self.dim, r=self.radius, noise=self.noise)
            sphere = origin + sphere
            spheres.append(sphere)
            labels.append(i * torch.ones(sphere.shape[0]))

        dataset = torch.cat(spheres)
        labels = torch.cat(labels)

        return dataset, labels

    def dsphere(self, n=100, d=2, r=1, noise=None):
        """
        Sample a d-sphere
        :param n: number of samples
        :param d: the intrinsic dimension of the sphere
        :param r: the sphere radius
        :param noise: whether to add noise
        :returns: spheres
        """

        # randomly generate data
        data = torch.randn(n, d + 1)

        # normalize data onto unit sphere
        normed_data = data / torch.norm(data, dim=1)[:, None]

        # scale unit sphere
        sphere_data = r * normed_data

        # add noise to data
        if noise:
            sphere_data += noise * torch.randn(*sphere_data.shape)

        return sphere_data


class Earth(CustomDataset):
    """
    Create an earth dataset
    """

    def __init__(self, filename=None, n_samples=0, *args, **kwargs):
        self.filename = filename
        super().__init__(*args, **kwargs)

        # dataset contains some weird labels, which I remove here
        self.dataset = self.dataset[self.labels != 6]
        self.n_samples -= torch.sum(self.labels == 6).item()
        self.labels = self.labels[self.labels != 6]

    @staticmethod
    def transform_labels(labels):
        string_labels = ["Africa", "Europe", "Asia", "North America", "Australia", "South America"]

        return string_labels

    def create(self):
        """
        Generate a swiss snail dataset.
        """

        data = torch.load(self.filename)

        # idx = torch.randperm(data.shape[0])[:self.n_samples]
        # data = data[idx]

        xs, ys, zs, labels = torch.unbind(data, dim=-1)
        dataset = torch.vstack((xs, ys, zs)).T.float()

        # dataset, labels = torch.unbind(data, dim=-1)
        # dataset = data[idx, :, :, :].float()
        # labels = data[idx, ]

        return dataset, labels

    def generate(self, n):
        """
        Generate and save the dataset
        """

        import geopandas

        bm = Basemap(projection="cyl")

        xs = []
        ys = []
        zs = []

        phis = []
        thetas = []

        # phi = long, theta = lat
        # das erste Argument is azimuth (phi), das zweite polar (theta) (in [-pi, pi])
        for phi in np.linspace(-180, 180, num=n):
            for theta in np.linspace(-90, 90, num=n):
                if bm.is_land(phi, theta):
                    phis.append(phi)
                    thetas.append(theta)

                    phi_rad = phi / 360 * 2 * np.pi
                    theta_rad = theta / 360 * 2 * np.pi

                    x = np.cos(phi_rad) * np.cos(theta_rad)
                    y = np.cos(theta_rad) * np.sin(phi_rad)
                    z = np.sin(theta_rad)

                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

        xs = torch.tensor(xs).float()
        ys = torch.tensor(ys).float()
        zs = torch.tensor(zs).float()

        # generate labels
        df = pd.DataFrame(
            {
                "longitude": phis,
                "latitude": thetas
            }
        )

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))

        results = geopandas.sjoin(gdf, world, how="left")

        le = preprocessing.LabelEncoder()
        encoded_results = torch.tensor(le.fit_transform(results["continent"].values))

        data = torch.vstack((xs, ys, zs, encoded_results)).T

        torch.save(data, self.filename)

        return data


class FigureEight(CustomDataset):
    """
    Create a figure-8 dataset
    """

    def create(self):
        """
        Generate a figure-8 dataset.
        """

        t = torch.rand(self.n_samples)

        t = torch.empty(self.n_samples).uniform_(0.01, 0.990)

        x = torch.sqrt(torch.tensor([2])) * torch.cos(2 * np.pi * t) / (torch.sin(2 * np.pi * t) ** 2 + 1)
        y = torch.sqrt(torch.tensor([2])) * torch.cos(2 * np.pi * t) * torch.sin(2 * np.pi * t) / (
                torch.sin(2 * np.pi * t) ** 2 + 1)

        X = torch.stack((x, y)).T

        return X, t


class Zilionis(CustomDataset):
    """
    Load the Zilionis dataset
    """

    def __init__(self, dir_path=None, n_samples=0, *args, **kwargs):
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(n_samples=n_samples, *args, **kwargs)

            mean_dataset = torch.mean(self.dataset, dim=1)
            std_dataset = torch.std(self.dataset, dim=1)
            self.dataset = (self.dataset - mean_dataset[:, None]) / std_dataset[:, None]

    # curtesy of https://github.com/hci-unihd/UMAPs-true-loss/blob/master/notebooks/UMAP_lung_cancer.ipynb
    def create(self):
        """
        Generate a figure-8 dataset.
        """

        pca306 = pd.read_csv(os.path.join(self.dir_path, "cancer_qc_final.txt"), sep='\t', header=None)
        pca306 = torch.tensor(pca306.to_numpy())

        meta = pd.read_csv(os.path.join(self.dir_path, "cancer_qc_final_metadata.txt"), sep="\t", header=0)

        cell_types = meta["Major cell type"].to_numpy()

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)

        # get only n samples
        # idx = torch.randperm(labels.shape[0])[:self.n_samples]
        # pca306 = pca306[idx]
        # labels = labels[idx]

        pca306 = pca306.float()
        labels = labels.float()

        return pca306, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(os.path.join(dir_path, "cancer_qc_final_metadata.txt"), sep="\t", header=0)
        cell_types = meta["Major cell type"].to_numpy()

        string_labels = np.unique(cell_types)
        string_labels = np.array([cell_type[1:] for cell_type in string_labels])

        return list(string_labels)


class PBMC(CustomDataset):
    """
    Load the PBMC dataset
    """

    def __init__(self, dir_path=None, n_samples=0, *args, **kwargs):
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(n_samples=n_samples, *args, **kwargs)

            # mean_dataset = torch.mean(self.dataset, dim=1)
            # std_dataset = torch.std(self.dataset, dim=1)
            # self.dataset = (self.dataset - mean_dataset[:, None]) / std_dataset[:, None]

    def create(self):
        """
        Generate a figure-8 dataset.
        """

        pca50 = pd.read_csv(os.path.join(self.dir_path, "pbmc_qc_final.txt"), sep='\t', header=None)
        pca50 = torch.tensor(pca50.to_numpy())

        meta = pd.read_csv(os.path.join(self.dir_path, "pbmc_qc_final_labels.txt"), sep="\t", header=None)

        cell_types = np.squeeze(meta.to_numpy())

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)

        # get only n samples
        # idx = torch.randperm(labels.shape[0])[:self.n_samples]
        # pca50 = pca50[idx]
        # labels = labels[idx]

        pca50 = pca50.float()
        labels = labels.float()

        return pca50, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(os.path.join(dir_path, "pbmc_qc_final_labels.txt"), sep="\t", header=None)
        cell_types = np.squeeze(meta.to_numpy())

        string_labels = np.unique(cell_types)
        string_labels = np.array([cell_type[1:] for cell_type in string_labels])

        return list(string_labels)


class CElegans(CustomDataset):
    """
    Load the C-Elegans dataset
    """

    def __init__(self, dir_path=None, n_samples=0, *args, **kwargs):
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(n_samples=n_samples, *args, **kwargs)

            # mean_dataset = torch.mean(self.dataset, dim=1)
            # std_dataset = torch.std(self.dataset, dim=1)
            # self.dataset = (self.dataset - mean_dataset[:, None]) / std_dataset[:, None]

    # curtesy of https://github.com/hci-unihd/UMAPs-true-loss/blob/master/notebooks/UMAP_lung_cancer.ipynb
    def create(self):

        pca100 = pd.read_csv(os.path.join(self.dir_path, "c-elegans_qc_final.txt"), sep='\t', header=None)
        meta = pd.read_csv(os.path.join(self.dir_path, "c-elegans_qc_final_metadata.txt"), sep=",", header=0)

        # remove instances where celltype is unknown
        meta["cell.type"] = meta["cell.type"].fillna("unknown")
        # print(meta["cell.type"][meta["cell.type"].isna()])
        # print(pca100[meta["cell.type"].isna()])
        # pca100 = pca100[meta["cell.type"].notna()]
        # meta = meta[meta["cell.type"].notna()]

        pca100 = torch.tensor(pca100.to_numpy())
        cell_types = meta["cell.type"].to_numpy()

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)

        # get only n samples
        # idx = torch.randperm(labels.shape[0])[:self.n_samples]
        # pca100 = pca100[idx]
        # labels = labels[idx]

        pca100 = pca100.float()
        labels = labels.float()

        return pca100, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(os.path.join(dir_path, "c-elegans_qc_final_metadata.txt"), sep=",", header=0)

        # remove instances where celltype is unknown
        meta["cell.type"] = meta["cell.type"].fillna("unknown")
        cell_types = meta["cell.type"].to_numpy()

        # labels = np.zeros(len(cell_types)).astype(int)
        # name_to_label = {}
        # label_to_name = {}
        # tests = []
        # for i, phase in enumerate(np.unique(cell_types)):
        #    tests.append(phase)
        #    label_to_name[i] = phase
        #    name_to_label[phase] = i
        #    labels[cell_types == phase] = i

        string_labels = np.unique(cell_types)
        # string_labels = np.array([cell_type[1:] for cell_type in string_labels])

        return list(string_labels)


class Images(CustomDataset):
    """
    Create an Image dataset
    """

    def __init__(self, img1_path, img2_path, *args, **kwargs):
        self.img1_path = img1_path
        self.img2_path = img2_path
        super().__init__(*args, **kwargs)

    def create(self):
        """
        Generate an Image dataset.
        """

        # x = np.arange(0, 500, 1)
        # y = np.arange(0, 340, 1)
        # z = np.zeros(500)
        x = np.linspace(0, 1, 500)
        y = np.linspace(0, 340 / 500, 340)
        # z = np.cos(2 * np.pi * np.linspace(0, 1, 500))
        # z = np.zeros(500)
        ampl = 0.2 * np.cos(2 * np.pi * np.linspace(0, 1, 500))
        z1 = np.linspace(0, 1, 500) + ampl
        z2 = np.linspace(-1, 0, 500) + ampl
        y = np.flip(y)

        x1 = x - 1
        x2 = x + 0.5
        y1 = y + 1
        y2 = y

        # z1 = z
        # z2 = z

        XX1, YY1 = np.meshgrid(x1, y1, indexing="ij")
        XX2, YY2 = np.meshgrid(x2, y2, indexing="ij")

        XX1 = np.repeat(XX1[:, :, np.newaxis], 1, axis=2)
        XX2 = np.repeat(XX2[:, :, np.newaxis], 1, axis=2)

        ZZ1 = np.repeat(z1[:, np.newaxis], 340, axis=1)
        ZZ1 = ZZ1[:, :, np.newaxis]
        ZZ2 = np.repeat(z2[:, np.newaxis], 340, axis=1)
        ZZ2 = ZZ2[:, :, np.newaxis]
        # ZZ1 = ZZ
        # ZZ2 = ZZ
        XX = np.stack([XX1, XX2])
        YY = np.stack([YY1, YY2])
        ZZ = np.stack([ZZ1, ZZ2])

        XX = XX.flatten()
        YY = YY.flatten()
        ZZ = ZZ.flatten()

        dataset = np.stack([XX, YY, ZZ]).T

        img1 = Image.open(self.img1_path)
        img2 = Image.open(self.img2_path)

        convert_tensor = transforms.ToTensor()

        img1 = convert_tensor(img1)
        img2 = convert_tensor(img2)

        img1 = torch.transpose(img1, 0, 2)
        img2 = torch.transpose(img2, 0, 2)

        img1 = img1.reshape(-1, 3)
        img2 = img2.reshape(-1, 3)

        img = torch.vstack([img1, img2])

        dataset = torch.from_numpy(dataset).float()

        return dataset, img
