# Load modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Some global data
global data
global X_all
global Y_all
global tlist


# ==================================================
# Helper functions
# ==================================================

def get_data(idx, data):
    """
    Get one training instance from the data set at the index idx

    Args:
        idx (int or array): An integer/array specifying which of the training example to fetch

    Returns:
        x (array): An array of shape (time_steps, 3) which specifies the input to
                   the neural network. The first column is the time and the second
                   and third columns specify the (x, y) coordinates of the second
                   particle. Note that the first particle is always assumed to be
                   at (1, 0) and the third particle can be inferred from the first
                   and second particle's position.

        y (array): An array of shape (time_steps, 4) which specifies the output that 
                   is expected from the neural network.

                   The first two columns specify the (x, y) coordinates of the first
                   particles and the next two columns give the coordinates of the 
                   second particle for the specified time (length of the columns).
                   The third particles position can be inferred from the first
                   and second particle's position.
    """
    if type(idx) != int:
        x = np.zeros((len(idx), len(data[idx[0]]), 3))
        for i in range(len(idx)):
            x[i, :, 0] = data[i, :, 0]
            x[i, :, 1] = data[i, 0, 3]  # only input the initial x_value
            x[i, :, 2] = data[i, 0, 4]  # only input the initial y_value
    else:
        x = np.zeros((len(data[idx]), 3))
        x[:, 0] = data[idx, :, 0]
        x[:, 1] = data[idx, 0, 3]  # only input the initial x_value
        x[:, 2] = data[idx, 0, 4]  # only input the initial y_value
    y = data[idx, :, 1:5]
    return x, y


def get_trajectories(pred):
    """
    Gets the trajectories from a predicted output pred.

    Args:
        pred (array): An array of shape (N, 4) where N is the number of time
                      steps. The four columns give the positions of the particles
                      1 and 2 for all the time steps.
    Returns:
        p1, p2, p3 (tuple of arrays): Three arrays of dimensions (N, 2) where N is the number 
                             of time steps and the two columns for each array give 
                             the positions of the three particles (p1, p2, p3)
    """
    x1 = pred[:, 0]
    y1 = pred[:, 1]
    x2 = pred[:, 2]
    y2 = pred[:, 3]

    x3 = - x1 - x2
    y3 = - y1 - y2
    return (x1, y1), (x2, y2), (x3, y3)


def plot_trajectories(p1, p2, p3, ax=None, **kwargs):
    """
    Plots trajectories for points p1, p2, p3

    Args:
        p1, p2, p3 (array): Three arrays each of shape (n, 2) where n is the number
                            of time steps. Each array is the (x, y) position for the
                            particles
        ax (axis object): Default None, in which case a new axis object is created.
        kwargs (dict): Optional keyword arguments for plotting

    Returns:
        ax: Axes object
    """
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(p1[0], p1[1], color='green', **kwargs)
    ax.plot(p2[0], p2[1], color='blue', **kwargs)
    ax.plot(p3[0], p3[1], color='red', **kwargs)
    return ax


def plot_index(idx, data, ax=None, **kwargs):
    """"""
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    x, y = get_data(idx, data)
    p1, p2, p3 = get_trajectories(y)
    ax = plot_trajectories(p1, p2, p3, ax=ax, **kwargs)
    return ax


def plot_history(hist, ax=None, **kwargs):
    """"""
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    epochs = np.arange(1, len(hist)+1)
    loss = hist[:, 2]
    ax.semilogy(epochs, loss, **kwargs)
    ax.set_ylabel(r"MAE")
    ax.set_xlabel(r"Epoch")
    ax.legend()
    return ax


def save_model(model, name):
    """Save the model weights to a .h5 file and the history (loss and accuracy)
    to a txt file. If a file already exists, then just append to it. 
    Args:
        model (keras.model): A keras model
        name (string): The name of the model, has no ending. (no .h5 or .txt)
    Returns:
        None."""
    model.save(name + '.h5')
    hist = model.history
    keys = [*hist.history.keys()]
    epochs = len(hist.history[keys[0]])
    header = ', '.join(keys)
    data = np.zeros((epochs, len(keys)))
    for i in range(len(keys)):
        data[:, i] = hist.history[keys[i]]
    fname = name + '.txt'
    if os.path.exists(fname):  # append to existing file
        prev_data = np.loadtxt(fname)
        data = np.append(prev_data, data, axis=0)
    np.savetxt(name + '.txt', data, header=header)


def load_model(name):
    """Load the model weights from a .h5 file and the history from a .txt file.
    Args:
        name (string): name of the files.
    Returns:
        model (keras.model): the model with the weights
        hist np.array: (loss, acc, test_loss, test_acc)the loss and accuracy 
                        as functions of the epoch."""
    model = tf.keras.models.load_model(name+'.h5')
    hist = np.loadtxt(name+'.txt')
    return model, hist


# ==================================================
# Load data
# also clear out stuck trajectories
# ==================================================

def check_collision_trajectory(idx, data):
    pos = data[idx, :, 1:5]
    x_zero = np.logical_and(pos[:, 0] == .0, pos[:, 1] == .0)
    y_zero = np.logical_and(pos[:, 2] == .0, pos[:, 3] == .0)
    zeros = np.logical_and(x_zero, y_zero)
    if np.sum(zeros > 0):
        #print("Stuck at 0 ?, index = ", idx)
        return idx
    else:
        return None


def load_data():
    """Load the data from the file 'data_project2.npz and return
    X and Y data without trajectories which end up stuck in a collision at 0.
    Args:
        None.
    Returns:
        X_all (np.array): input data of the shape (N, 3) with t, x_init, y_init
        Y_all (np.array): target data of the shape (N, 4), with x1,y1, x2,y2
        tlist (tf.Tensor): time data as a tf.Tensor object"""
    load_data = np.load('data_project2.npz')
    data = load_data['arr_0']

    # filter out the collision tajectories

    zero_traj_idx = []
    for i in range(len(data)):
        ind = check_collision_trajectory(i, data)
        if ind != None:
            zero_traj_idx.append(ind)

    data = np.delete(data, np.array(zero_traj_idx), 0)
    return data


data = load_data()
X_all, Y_all = get_data(np.arange(0, len(data)), data)
perm = np.random.permutation(range(len(X_all)))
X_all = X_all[perm]
Y_all = Y_all[perm]

# for the custom loss function
tlist = data[0, :, 0].reshape((-1, 1))
tlist = tf.convert_to_tensor(tlist.astype(np.float32))
split = int(0.9*len(perm))


# ==================================================
# Custom loss tensorflow function
# ==================================================

@tf.function()
def tf_compute_velocities(t, p):
    """Computes the velocities of the particles from the trajectories starting from rest.
    The first velocity is simply left out, since it is always 0 and doesnt contribute to the 
    kinetic energy.

    Args:
        t (array[float]): An array of shape (N, 1) giving time steps.
        p (array): An array of dimension (N, 2) where N is the number 
                   of time steps and the two columns give the coordinates
                   of the particle.

    Returns:
        v (array): An array of dimension (N-1, 2) where N is the number 
                   of time steps and the two columns for each array give 
                   the x and y components for the instantaneous velocity
                   of the particle. """
    v = (p[1:]-p[:-1])/(t[1:]-t[:-1])
    return v


@tf.function()
def tf_compute_kinetic_energy(v):
    """ Computes the kinetic energy for the given velocity vectors
    Args:
        v (array): A (N,2) array of veolcities for N time steps.
    Returns:
        ke (array): An array of shape (N, 1) giving the kinetic energies at each time step."""
    return tf.reshape(.5 * tf.reduce_sum(v**2, 1), (-1, 1))


@tf.function()
def tf_compute_potential_energy(p1, p2, p3):
    """Computes the potential energy for the given position vectors. The value of the 
    gravitational constant is taken as 1 (G=1). The masses are the same value (m=1)
    Args:
        p1, p2, p3 (arrays): Three arrays of dimensions (N, 2) where N is the number 
                             of time steps and the two columns for each array give 
                             the positions of the three particles (p1, p2, p3)
    Returns
        pe (array): An array of shape(N, 1) giving the potential energy at each time step

    """
    d_12 = tf.sqrt((p1[:, 0] - p2[:, 0])**2 + (p1[:, 1]-p2[:, 1])**2)
    d_13 = tf.sqrt((p1[:, 0] - p3[:, 0])**2 + (p1[:, 1]-p3[:, 1])**2)
    d_23 = tf.sqrt((p3[:, 0] - p2[:, 0])**2 + (p3[:, 1]-p2[:, 1])**2)
    pe = -1 * (1/d_12 + 1/d_13 + 1/d_23)
    return tf.reshape(pe, (-1, 1))


@tf.function()
def custom_loss(y, y_pred):
    """
    A custom loss function computing error in energy conservation.
    """
    predicted_positions1 = y_pred[:, 0:2]
    predicted_positions2 = y_pred[:, 2:4]
    predicted_positions3 = -y_pred[:, 0:2] - y_pred[:, 2:4]

    predicted_velocities1 = tf_compute_velocities(tlist, predicted_positions1)
    predicted_velocities2 = tf_compute_velocities(tlist, predicted_positions2)
    predicted_velocities3 = tf_compute_velocities(tlist, predicted_positions3)

    initial_potential_energy = -1/tf.sqrt((predicted_positions1[0, 0] -
                                           predicted_positions2[0, 0])**2
                                          + (predicted_positions1[0, 1] -
                                             predicted_positions2[0, 1])**2)
    initial_potential_energy += -1/tf.sqrt((predicted_positions1[0, 0] -
                                            predicted_positions3[0, 0])**2
                                           + (predicted_positions1[0, 1] -
                                              predicted_positions3[0, 1])**2)
    initial_potential_energy += -1/tf.sqrt((predicted_positions3[0, 0] -
                                            predicted_positions2[0, 0])**2
                                           + (predicted_positions3[0, 1] -
                                              predicted_positions2[0, 1])**2)

    ke_predicted_trajectory = tf_compute_kinetic_energy(predicted_velocities1)
    ke_predicted_trajectory += tf_compute_kinetic_energy(predicted_velocities2)
    ke_predicted_trajectory += tf_compute_kinetic_energy(predicted_velocities3)

    # only compute the potential energy from second step, the first one is always 0
    pe_predicted_trajectory = tf_compute_potential_energy(predicted_positions1[1:],
                                                          predicted_positions2[1:],
                                                          predicted_positions3[1:])

    error = ke_predicted_trajectory + pe_predicted_trajectory - initial_potential_energy

    error = tf.clip_by_value(error, -1e6, 1e6),
    energy_loss = tf.reduce_mean(tf.abs(error))

    # The relative weight ofthe two terms in the custom loss might be tuned.
    return tf.keras.losses.MeanAbsoluteError()(y, y_pred) + 0.001*energy_loss
