import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pylab

from utils.misc import create_directory

# Changing plotting parameters
# - https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
# ----------------------------


font_size = 30  # 'xx-large'
# params = {'legend.fontsize': font_size,
#           'figure.figsize': (18, 5.5),
#           'axes.labelsize': font_size,
#           'axes.titlesize': font_size,
#           'xtick.labelsize': font_size,
#           'ytick.labelsize': font_size}
params = {'font.size': font_size}
pylab.rcParams.update(params)
pylab.rcParams['axes.linewidth'] = 3.5


def plot_federated_clients_distribution(clients_datasets_list, batch_size, base_save_path=None):
    sizes = []
    num_clients = len(clients_datasets_list)

    for ds in clients_datasets_list:
        sizes.append(len(list(ds.as_numpy_iterator())) * batch_size)

    plt.figure(figsize=(10, 7))
    plt.bar(np.arange(len(sizes)), sizes)
    plt.xticks(range(len(sizes)), range(1, len(sizes) + 1))
    plt.ylim(bottom=np.min(sizes) - 500, top=np.max(sizes) + 500)
    plt.xlabel("client")
    plt.ylabel("# traffic matrices")
    if base_save_path:
        create_directory(base_save_path)
        save_path = os.path.join(base_save_path, f"clients_distribution_n{num_clients}.png")
        plt.savefig(save_path)
    plt.show()


def plot_federated_clients_mean_matrix(clients_datasets_list, base_save_path=None):
    num_clients = len(clients_datasets_list)

    mean_matrices_list = []

    for i, ds in enumerate(clients_datasets_list):
        mean_matrix = np.mean(np.array(list(ds.unbatch().as_numpy_iterator()))[:, 1])[:, :, 0]
        mean_matrices_list.append(mean_matrix)

        if base_save_path:
            create_directory(base_save_path)
            save_path = os.path.join(base_save_path, f"clients_mean_matrix_{i + 1}.png")
        else:
            save_path = base_save_path

        plot_tm(mean_matrix, max_val=1., colorbar=True, save_path=save_path)

    return mean_matrices_list


def plot_tm(tm, max_val=1, colorbar=True, save_path=None, title=None, xlabel=None, ylabel=None, cbarlabel=None):
    # ----------------------------

    tm = np.squeeze(tm)

    # Use the following for the thesis
    #axes_ticks = np.arange(0, tm.shape[0], np.min([tm.shape[0] // 3, 5]))
    # Use the following for the paper
    axes_ticks = np.arange(0, tm.shape[0], 10)

    bkp_xtick_bottom = plt.rcParams['xtick.bottom']
    bkp_xtick_top = plt.rcParams['xtick.top']

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.figure(figsize=(12, 12))
    plt.imshow(tm, cmap="jet")
    # Set the range of the values and its mapping to plot colors. Without the following line the "highest" color
    #   will be associated to the highest number in the input matrix, also if it's different from `max_val`.
    if max_val:
        plt.clim(0, max_val)

    plt.xticks(axes_ticks, axes_ticks)
    plt.yticks(axes_ticks, axes_ticks)

    if title:
        plt.title(title)

    #     if xlabel:
    #         plt.xlabel(xlabel)

    #     if ylabel:
    #         plt.ylabel(ylabel)

    if colorbar:
        #         cbar = plt.colorbar(orientation="horizontal")
        cbar = plt.colorbar(orientation="horizontal", fraction=0.046, pad=0.04)
        #         cbar.ax.set_xticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()]) # set ticks of your format
        print(cbar.get_ticks())
        if cbarlabel:
            cbar.set_label(cbarlabel)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = bkp_xtick_bottom
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = bkp_xtick_top


def plot_lr_hr(lr, hr, colorbar=True):
    x = np.arange(0, lr.shape[0] + 1, 1)
    y = np.arange(0, lr.shape[0] + 1, 1)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.subplot(1, 2, 1)
    plt.title("Low Resolution")
    plt.pcolor(X, Y, lr, vmin=0, vmax=1, cmap="jet")  # ,cmap="gray"

    x = np.arange(0, hr.shape[0] + 1, 1)
    y = np.arange(0, hr.shape[0] + 1, 1)
    X, Y = np.meshgrid(x, y)
    plt.subplot(1, 2, 2)
    plt.title("High Resolution")
    plt.pcolor(X, Y, hr, vmin=0, vmax=1, cmap="jet")
    plt.subplots_adjust(left=0.07, right=0.86)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(cax=cax)  # fraction=0.046, pad=0.04
    plt.show()


def plot_lr_hr_sr(model, lr_val, hr_val, max_val, num_tests=1, base_save_path=None, dark_mode=False):
    sr_val = tf.squeeze(model.predict(lr_val[:num_tests]))
    create_directory(base_save_path)
    for counter in range(num_tests):

        print("Test #{}/{}\n".format(counter+1, num_tests))

        sr = sr_val[counter]
        lr = tf.squeeze(lr_val[counter])
        hr = tf.cast(tf.squeeze(hr_val[counter]), dtype=tf.float32)

        if sr.shape != hr.shape:
            raise ValueError('SR shape must be equal to HR shape')

        scale_factor = hr.shape[0] // lr.shape[0]

        bic = tf.squeeze(
            tf.image.resize(lr_val[counter], [lr.shape[0] * scale_factor, lr.shape[1] * scale_factor], method="bicubic",
                            antialias=True)).numpy()

        print("LR shape: \t", lr.shape)
        print("HR shape: \t", hr.shape)
        print("SR shape: \t", sr.shape)
        print("BIC shape: \t", bic.shape)

        mae_sr = np.mean(np.abs(sr - hr))
        print("\nMAE_SR: {:.4f} --> {:.4f} kbps".format(mae_sr, mae_sr * max_val))
        print("MIN_SR error: {:.4f} --> {:.4f} kbps".format(np.min(np.abs(sr - hr)), np.min(np.abs(sr - hr)) * max_val))
        print("MAX_SR error: {:.4f} --> {:.4f} kbps".format(np.max(np.abs(sr - hr)), np.max(np.abs(sr - hr)) * max_val))

        mae_bic = np.mean(np.abs(bic - hr))
        print("\nMAE_BIC: {:.4f} --> {:.4f} kbps".format(mae_bic, mae_bic * max_val))
        print("MIN_BIC error: {:.4f} --> {:.4f} kbps".format(np.min(np.abs(bic - hr)),
                                                             np.min(np.abs(bic - hr)) * max_val))
        print("MAX_BIC error: {:.4f} --> {:.4f} kbps".format(np.max(np.abs(bic - hr)),
                                                             np.max(np.abs(bic - hr)) * max_val))

        if base_save_path:
            plot_tm(hr, colorbar=True, save_path=os.path.join(base_save_path, "hr_{:04}.png".format(counter)))
            plot_tm(lr, colorbar=True, save_path=os.path.join(base_save_path, "lr_{:04}.png".format(counter)))
            plot_tm(sr, colorbar=True, save_path=os.path.join(base_save_path, "sr_{:04}.png".format(counter)))
            plot_tm(bic, colorbar=True, save_path=os.path.join(base_save_path, "bic_{:04}.png".format(counter)))
