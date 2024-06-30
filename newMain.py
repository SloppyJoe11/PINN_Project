import os
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow C++ warnings


def build_pinn_model(input_shape=2, num_neurons=100, num_layers=4, output_shape=2):
    inputs = Input(shape=(input_shape,))
    x = Dense(num_neurons, activation='tanh')(inputs)
    for _ in range(num_layers - 1):
        x = Dense(num_neurons, activation='tanh')(x)
    outputs = Dense(output_shape, activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def standardize_data(data):
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    standardized_data = (data - mean_val) / std_val
    return standardized_data

optimizer = tf.keras.optimizers.Adam()

# Parameters for the NLSE
beta_2 = -20  # ps^2/km
gamma = 1.3  # (V*km)^-1
alpha = 0  # m^-1 (setting to zero to remove attenuation)

# parameters for data
fiber_length_Km = 500
dz_m = 100
pulse_width_ps = 50 * 1e-12
dt_ps = pulse_width_ps / 10
num_steps = 256


Z = np.arange(0, fiber_length_Km*1e3, dz_m)
T = np.arange(-num_steps // 2, num_steps // 2) * dt_ps   # Time vector in ps

# Create a 2D grid of Z and T values
Z_grid, T_grid = np.meshgrid(Z, T, indexing='ij')
input_data = np.vstack((Z_grid.flatten(), T_grid.flatten())).T


# Apply standardization
standardized_input_data = standardize_data(input_data)


# Create the dataset, shuffle, and batch it
batch_size = 32
buffer_size = len(standardized_input_data)  # Typically, the buffer size is set to the size of the dataset


def train_loss(model, train_batch):
    z, t = tf.split(train_batch, [1, 1], axis=-1)

    with tf.GradientTape() as tape2:
        tape2.watch(z)
        a_pred =




def train_step(model, train_batch):
    with tf.GradientTape() as tape:
        loss = train_loss(model, train_batch)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



if __name__ == "__main__":

    epochs = 30
    for epoch in range(epochs):

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        train_batch_num = 1

        train_dataset = tf.data.Dataset.from_tensor_slices(standardized_input_data)
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

        for batch in train_dataset:
            loss = train_step(model, batch)
            pass
