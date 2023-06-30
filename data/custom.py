"""
Create Custom Datasets
"""

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
        Base Class for a Custom Dataset
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

        if self.train:
            self.n_samples = len(self.dataset)
        else:
            self.n_samples = test_size
            self.dataset = self.dataset[test_subset.indices]
            self.labels = self.labels[test_subset.indices]

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
        xs, ys, zs, labels = torch.unbind(data, dim=-1)
        dataset = torch.vstack((xs, ys, zs)).T.float()

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

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth'))
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))

        results = geopandas.sjoin(gdf, world, how="left")

        le = preprocessing.LabelEncoder()
        encoded_results = torch.tensor(le.fit_transform(results["continent"].values))

        data = torch.vstack((xs, ys, zs, encoded_results)).T

        torch.save(data, self.filename)

        return data


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

        pca50 = pca50.float()
        labels = labels.float()

        return pca50, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(os.path.join(dir_path, "pbmc_qc_final_labels.txt"), sep="\t", header=None)
        cell_types = np.squeeze(meta.to_numpy())

        string_labels = np.unique(cell_types)

        return list(string_labels)


class CElegans(CustomDataset):
    """
    Load the C-Elegans dataset
    """

    def __init__(self, dir_path=None, n_samples=0, *args, **kwargs):
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(n_samples=n_samples, *args, **kwargs)

    def create(self):
        pca100 = pd.read_csv(os.path.join(self.dir_path, "c-elegans_qc_final.txt"), sep='\t', header=None)
        meta = pd.read_csv(os.path.join(self.dir_path, "c-elegans_qc_final_metadata.txt"), sep=",", header=0)

        # remove instances where celltype is unknown
        meta["cell.type"] = meta["cell.type"].fillna("unknown")

        pca100 = torch.tensor(pca100.to_numpy())
        cell_types = meta["cell.type"].to_numpy()

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)

        pca100 = pca100.float()
        labels = labels.float()

        return pca100, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(os.path.join(dir_path, "c-elegans_qc_final_metadata.txt"), sep=",", header=0)

        # remove instances where celltype is unknown
        meta["cell.type"] = meta["cell.type"].fillna("unknown")
        cell_types = meta["cell.type"].to_numpy()

        string_labels = np.unique(cell_types)
        return list(string_labels)


class PBMC_new(CustomDataset):
    """
    Load the PBMC dataset
    """

    def __init__(self, dir_path=None, n_samples=0, *args, **kwargs):
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(n_samples=n_samples, *args, **kwargs)

    def create(self):
        pca50 = np.load(os.path.join(self.dir_path, "pca50.npy"))
        pca50 = torch.from_numpy(pca50)

        meta = pd.read_csv(os.path.join(self.dir_path, "zheng17-cell-labels.txt"), sep="\t", header=None, skiprows=1)
        meta = meta.to_numpy()[:, 1]
        cell_types = np.squeeze(meta)

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)

        pca50 = pca50.float()
        labels = labels.float()

        return pca50, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(os.path.join(dir_path, "zheng17-cell-labels.txt"), sep="\t", header=None, skiprows=1)
        meta = meta.to_numpy()[:, 1]

        cell_types = np.squeeze(meta)

        string_labels = np.unique(cell_types)

        return list(string_labels)
