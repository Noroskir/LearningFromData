import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

from project import *

import config


parser = argparse.ArgumentParser(description='3-Body Newton Neural Net')
parser.add_argument('-o', '--output')  # the name to which the model gets saved
parser.add_argument('-e', '--epochs')  # number of epochs to train the model
parser.add_argument('-f', '--file')  # file for the hyperparameters
parser.add_argument('-b', '--batchsize')
parser.add_argument('-q', '--quiet', action='store_true')


args = parser.parse_args()
batchsize = int(args.batchsize)
epochs = int(args.epochs)

conf = config.Config(args.file)
conf_args = conf.parse_config_args()
print(conf_args)

eta = conf_args['eta']
beta1 = conf_args['beta1']
beta2 = conf_args['beta2']
lambda_energy = conf_args['lambda']


if not args.quiet:
    print("Train model {:} with the configuration:".format(args.output))
    print(f"eta:           {eta:.5g}")
    print(f"beta1:         {beta1:.5g}")
    print(f"beta2:         {beta2:.5g}")
    print(f"lambda_energy: {lambda_energy:.5g}")
    print(f"Epochs:        {epochs:}")
    print(f"Batchsize:     {batchsize:}")


# ==================================================
# Load data and create batches
# ==================================================

train_dataset = tf.data.Dataset.from_tensor_slices((X_all[:split].reshape((-1, 3)),
                                                    Y_all[:split].reshape((-1, 4))))
train_dataset = train_dataset.batch(batchsize)

test_dataset = tf.data.Dataset.from_tensor_slices((X_all[split:].reshape((-1, 3)),
                                                   Y_all[split:].reshape((-1, 4))))
test_dataset = test_dataset.batch(batchsize)


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(3),
    tf.keras.layers.Dense(
        128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),  # 1
    tf.keras.layers.Dense(128, activation='relu'),  # 2
    tf.keras.layers.Dense(
        128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),  # 3
    tf.keras.layers.Dense(128, activation='relu'),  # 4
    tf.keras.layers.Dense(
        128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),  # 5
    tf.keras.layers.Dense(128, activation='relu'),  # 6
    tf.keras.layers.Dense(
        128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),  # 7
    tf.keras.layers.Dense(128, activation='relu'),  # 8
    tf.keras.layers.Dense(
        128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),  # 9
    tf.keras.layers.Dense(128, activation='relu'),  # 10
    tf.keras.layers.Dense(4, activation='linear')])


model.compile(optimizer=keras.optimizers.Adam(eta, beta1, beta2),
              loss=custom_loss,
              metrics=['accuracy'])

hist = model.fit(train_dataset, epochs=epochs,
                 validation_data=test_dataset)

save_model(model, args.output)
