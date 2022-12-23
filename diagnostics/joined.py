from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from conf import device
from data.handle_data import data_forward
from util import get_sc_kwargs


def common_plot(test_loader, model, subplots, writer=None):
    """
    Create common plot from the given subplots
    :param test_loader: the dataloader for the test data
    :param model: the considered model
    :param subplots: 2D array containing the subplots
    :param writer: SummaryWriter object
    :return:
    """

    inputs, outputs, latent_activations, labels = data_forward(model.to(device), test_loader)

    """
    PLOTTING
    """
    latent_activations = latent_activations.detach().cpu()

    sc_kwargs = get_sc_kwargs()

    fig, axs = plt.subplots(3, len(subplots[0]))
    # fig.suptitle(f"Evaluation of {model.__class__.__name__}")

    axs[0, 1].scatter(latent_activations[:, 0], latent_activations[:, 1], **sc_kwargs, c=labels, cmap="tab10co")
    axs[0, 1].set_aspect("equal")
    # axs[0, 1].set_title("Latent Space")

    axs[0, 0].axis("off")
    axs[0, 2].axis("off")

    for i, row in enumerate(subplots):
        for j, column in enumerate(row):
            obj = subplots[i][j]

            if type(obj) == tuple:
                subplot, collection = obj

                divider = make_axes_locatable(axs[i + 1, j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(collection, cax=cax)

            else:
                subplot = obj

            # erste Reihe überspringen
            position = axs[i + 1, j].get_position()
            axs[i + 1, j].remove()

            subplot.figure = fig
            fig.axes.append(subplot)
            fig.add_axes(subplot)

            subplot.set_position(position)

    writer.add_figure("summary", fig)

    plt.show()

    writer.flush()
    writer.close()
