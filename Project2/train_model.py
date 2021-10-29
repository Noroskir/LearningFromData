import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

from project import *

import config

global lambda_energy


parser = argparse.ArgumentParser(description='3-Body Newton Neural Net')
parser.add_argument('-o', '--output')  # the dir to which the model gets saved
parser.add_argument('-f', '--file')  # file for the hyperparameters
parser.add_argument('-q', '--quiet', action='store_true')


args = parser.parse_args()

conf = config.Config(args.file)
conf_args = conf.parse_config_args()

eta = conf_args['eta']
beta1 = conf_args['beta1']
beta2 = conf_args['beta2']
lambda_energy = conf_args['lambda']
epochs = conf_args['epochs']
batchsize = conf_args['batchsize']
loss = conf_args['loss_function']

if loss == "MAE":
    loss_func = tf.keras.losses.MeanAbsoluteError()
if loss == "custom":
    loss_func = custom_loss


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

# ==================================================
# check if pretrained model exists or create model
# ==================================================

if os.path.exists(args.output + '.h5'):
    if loss == 'custom':
        model = keras.models.load_model(args.output + '.h5',
                                        custom_objects={'custom_loss': custom_loss})
    else:
        model = keras.models.load_model(args.output + '.h5')
else:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(3),
        tf.keras.layers.Dense(128, activation='relu'),  # 1
        tf.keras.layers.Dense(128, activation='relu'),  # 2
        tf.keras.layers.Dense(128, activation='relu'),  # 3
        tf.keras.layers.Dense(128, activation='relu'),  # 4
        tf.keras.layers.Dense(128, activation='relu'),  # 5
        tf.keras.layers.Dense(128, activation='relu'),  # 6
        tf.keras.layers.Dense(128, activation='relu'),  # 7
        tf.keras.layers.Dense(128, activation='relu'),  # 8
        tf.keras.layers.Dense(128, activation='relu'),  # 9
        tf.keras.layers.Dense(128, activation='relu'),  # 10
        tf.keras.layers.Dense(4, activation='linear')])


model.compile(optimizer=keras.optimizers.Adam(eta, beta1, beta2),
              loss=loss_func,
              metrics=['accuracy'])

hist = model.fit(train_dataset, epochs=epochs,
                 validation_data=test_dataset)

save_model(model, args.output)
