import os  # nopep8
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # nopep8

from project import *

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np


plt.rc('axes', titlesize=16, labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc("legend", fontsize=14)


#model, hist = load_model("Models/final/model")
model = tf.keras.models.load_model("big_model.h5")
energy_model, energy_hist = load_model("Models/energy_final/model",
                                       custom_objects={'custom_loss': custom_loss})
pre_trained_model = keras.models.load_model("Breen_NN_project2.h5")


def plot_fig3(hist, save=False):
    """"""
    fig, ax = plt.subplots(1, 1)
    epochs = np.arange(1, len(hist)+1)
    loss = hist[:, 2]
    ax.semilogy(epochs, loss)
    ax.set_ylabel(r"MAE")
    ax.set_xlabel(r"Epoch")
    if save:
        plt.savefig("../../Tex/Project2/figures/fig3.pdf")
    plt.show()


def plot_fig4(model, idxs, save=False):
    """"""
    fig, axes = plt.subplots(4, 2)
    axes[0][0].set_title("Our ANN")
    axes[0][1].set_title("Pre-Trained ANN")
    for i in range(len(axes)):
        xdata, ydata = get_data(idxs[i], data)
        pred = model.predict(xdata)
        p1, p2, p3 = get_trajectories(pred)
        axes[i][0] = plot_trajectories(p1, p2, p3, ax=axes[i][0])
        pred = pre_trained_model.predict(xdata)
        p1, p2, p3 = get_trajectories(pred)
        axes[i][1] = plot_trajectories(p1, p2, p3, ax=axes[i][1])
        # get real trajectories
        p1, p2, p3 = get_trajectories(ydata)
        axes[i][0] = plot_trajectories(p1, p2, p3, ax=axes[i][0], alpha=0.4)
        axes[i][1] = plot_trajectories(p1, p2, p3, ax=axes[i][1], alpha=0.4)
        axes[i][0].tick_params(labelbottom=False)
        axes[i][1].tick_params(labelleft=False, labelbottom=False)

    axes[-1][0].tick_params(labelleft=True, labelbottom=True)
    axes[-1][1].tick_params(labelbottom=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    if save:
        plt.savefig("../../Tex/Project2/figures/fig4.pdf")
    plt.show()


def plot_fig5(model, pre_trained_model, save=False):
    # create 1000 starting points for the trajectory

    times = [0.0, 0.95, 1.9, 2.95, 3.8]
    alphas = np.random.uniform(0, 2*np.pi, 1000)
    radii = np.random.uniform(0, 0.01, 1000)

    # generate input data:

    data_inp = np.zeros((1000, len(times), 3))  # N, t, (t, x, y)

    t = data[0, :, 0].reshape((-1, 1))
    x = radii*np.cos(alphas) + -.2
    y = radii*np.sin(alphas) + 0.3

    # find out the indices for the times

    full_inp = np.zeros((1000, 3))
    full_inp[:, 0] = t.flatten()
    full_inp[:, 1] = -.2
    full_inp[:, 2] = .3

    for i in range(len(times)):
        data_inp[:, i, 0] = t[np.argmin(np.abs(t - times[i]))]
        print("input time is: ", t[np.argmin(np.abs(t - times[i]))])
        data_inp[:, i, 1] = x
        data_inp[:, i, 2] = y

    data_inp = np.reshape(data_inp, (-1, 3))

    y_err = model.predict(data_inp)
    pred_full = model.predict(full_inp)

    p1, p2, p3 = get_trajectories(pred_full)

    fig, ax = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.0)

    ax[0].plot(p2[0], p2[1], '--', label='our ANN', alpha=0.3)
    ax[0].plot(y_err[:, 2], y_err[:, 3], 'o', ms=0.2, color="blue")
    ax[0].legend()
    ax[0].tick_params(labelbottom=False)

    y_err = pre_trained_model.predict(data_inp)
    pred_full = pre_trained_model.predict(full_inp)

    p1, p2, p3 = get_trajectories(pred_full)
    ax[1].plot(p2[0], p2[1], '--', label='pre-trained ANN', alpha=0.3)
    ax[1].plot(y_err[:, 2], y_err[:, 3], 'o', ms=0.2, color="blue")
    ax[1].legend()

    if save:
        plt.savefig("../../Tex/Project2/figures/fig5.pdf")
    plt.show()


def plot_fig6(model, energy_model, idx, save=False):
    """"""
    fig, ax = plt.subplots(2, 1, figsize=(7, 9))
    plt.subplots_adjust(hspace=0.0)
    for i in range(2):
        X, Y = get_data(idx[i], data)
        pred_energy = energy_model.predict(X)
        error_energy = energy_error(Y, pred_energy)

        pred_pre = pre_trained_model.predict(X)
        error_pre = energy_error(Y, pred_pre)

        pred = model.predict(X)
        error = energy_error(Y, pred)

        ax[i].semilogy(tlist[1:], error, label="MAE loss", color='orange')
        ax[i].semilogy(tlist[1:], error_energy,
                       label="Modified loss", color='#1f77b4')
        #ax[i].semilogy(tlist[1:], error_pre, label="Pre trained", alpha=0.4)
        ax[i].legend()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False,
                    bottom=False, left=False, right=False)

    plt.ylabel("rel. energy error")
    ax[1].set_xlabel("Time")
    ax[0].tick_params(labelbottom=False)
    if save:
        plt.savefig("../../Tex/Project2/figures/fig6.pdf")
    plt.show()


# plot_fig3(hist)
# plot_fig4(model, [0, 1, 2, 3])
# plot_fig5(model, pre_trained_model)
plot_fig6(model, energy_model, [0, 1])
