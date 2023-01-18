"""Competitor dimensionality reduction algorithms."""
from keras.regularizers import l2
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from umap import UMAP

import tensorflow as tf

from umap.parametric_umap import ParametricUMAP as ParametricUMAP_vanilla

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE

except ImportError:
    from sklearn.manifold import TSNE


class ParametricUMAP(ParametricUMAP_vanilla):
    def __init__(self,
                 min_dist=0.1,
                 n_components=2,
                 n_neighbors=15,
                 autoencoder_loss=True,
                 parametric_reconstruction=True,
                 input_dim=784,
                 rundir="",
                 *args, **kwargs):
        """

        """

        parametric_reconstruction_loss_fcn = tf.keras.losses.MeanSquaredError(
            # reduction=tf.keras.losses_utils.ReductionV2.AUTO,
            # reduction="auto",
            name='mean_squared_error'
        )
        latent_dim = 2

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_dim),
                tf.keras.layers.Activation("elu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(units=latent_dim, name="z"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=latent_dim),
                tf.keras.layers.Activation("elu"),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(units=100, activation="elu", kernel_regularizer=l2(1e-5)),
                tf.keras.layers.Dense(
                    units=input_dim, name="recon", activation=None
                ),
                # tf.keras.layers.Reshape(dims),
            ]
        )

        super().__init__(encoder=self.encoder,
                         decoder=self.decoder,
                         autoencoder_loss=autoencoder_loss,
                         parametric_reconstruction=parametric_reconstruction,
                         parametric_reconstruction_loss_fcn=parametric_reconstruction_loss_fcn,
                         min_dist=min_dist,
                         n_components=n_components,
                         n_neighbors=n_neighbors,
                         # batch_size=125
                         )
