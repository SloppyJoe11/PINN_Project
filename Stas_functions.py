import os
import sys
import time
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

# Set the backend for Matplotlib to 'Agg'
import matplotlib
matplotlib.use('Agg')


# Build the model
def build_pinn_model(input_shape=2, num_neurons=100, num_layers=4, output_shape=2):
    inputs = Input(shape=(input_shape,))
    x = Dense(num_neurons, activation='tanh')(inputs)
    for _ in range(num_layers - 1):
        x = Dense(num_neurons, activation='tanh')(x)
    outputs = Dense(output_shape, activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Plot training history
def plot_history(history):
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.plot(history['test_loss'], label='Test')
    plt.plot(history['nlse_loss'], label='NLSE')
    plt.plot(history['A0_loss'], label='A0')
    plt.plot(history['Ab_loss'], label='Ab')
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig('History plot')
    plt.close()


# Train loss function
def train_loss(fiber_batch, A0_batch, boundary_batch, pinn_model, parameters, nlse_avg, a0_avg, ab_avg):

    nlse_loss_term = nlse_residual(fiber_batch, pinn_model, parameters)
    nlse_avg.update_state(nlse_loss_term)

    A0_mse_term = initial_condition_loss(A0_batch, pinn_model)
    a0_avg.update_state(A0_mse_term)

    Ab_mse_term = boundary_condition_loss(boundary_batch, pinn_model)
    ab_avg.update_state(Ab_mse_term)

    return nlse_loss_term + A0_mse_term + Ab_mse_term


def nlse_residual(fiber_batch, pinn_model, parameters):
    z, t, a_real, a_image = tf.split(fiber_batch, [1, 1, 1, 1], axis=-1)
    alpha = parameters['alpha']
    beta_2 = parameters['beta_2']
    gamma = parameters['gamma']

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([z, t])
        a_pred = pinn_model(tf.concat([z, t], axis=-1))
        a_pred_real, a_pred_imag = tf.split(a_pred, 2, axis=-1)
        a_z_real = tape2.gradient(a_pred_real, z)
        a_z_imag = tape2.gradient(a_pred_imag, z)
        a_t_real = tape2.gradient(a_pred_real, t)
        a_t_imag = tape2.gradient(a_pred_imag, t)

    a_pred_complex = tf.complex(a_pred_real, a_pred_imag)
    a_z_complex = tf.cast(tf.complex(a_z_real, a_z_imag), tf.complex64)

    a_tt_real = tape2.gradient(a_t_real, t)
    a_tt_imag = tape2.gradient(a_t_imag, t)
    a_tt_complex = tf.complex(a_tt_real, a_tt_imag)

    del tape2

    a_pred_abs_squared = tf.cast(tf.square(tf.abs(a_pred_complex)), tf.complex64)
    attenuation_complex = (alpha / 2) * a_pred_complex
    chrom_dis_complex = (1j * beta_2 / 2) * tf.cast(a_tt_complex, tf.complex64)
    non_lin_complex = 1j * gamma * a_pred_abs_squared * a_pred_complex

    nlse_residual_value = a_z_complex + chrom_dis_complex + attenuation_complex - non_lin_complex
    nlse_term = tf.reduce_mean(tf.square(tf.abs(nlse_residual_value)))

    return nlse_term


def initial_condition_loss(A0_batch, pinn_model):
    z0, t0, A0_real, A0_image = tf.split(A0_batch, [1, 1, 1, 1], axis=-1)
    A0_ssfm = tf.cast(tf.complex(A0_real, A0_image), tf.complex64)
    A0_pred = pinn_model(tf.concat([z0, t0], axis=1))
    A0_pred_real, A0_pred_image = tf.split(A0_pred, 2, axis=-1)
    A0_pred_complex = tf.complex(A0_pred_real, A0_pred_image)
    A0_mse = tf.reduce_mean(tf.square(tf.abs(A0_pred_complex - A0_ssfm)))
    return A0_mse


def boundary_condition_loss(boundary_batch, pinn_model):
    zb, tb, Ab_real, Ab_image = tf.split(boundary_batch, [1, 1, 1, 1], axis=-1)
    Ab_ssfm = tf.cast(tf.complex(Ab_real, Ab_image), tf.complex64)
    Ab_pred = pinn_model(tf.concat([zb, tb], axis=-1))
    Ab_pred_real, Ab_pred_imag = tf.split(Ab_pred, 2, axis=-1)
    Ab_pred_complex = tf.complex(Ab_pred_real, Ab_pred_imag)
    Ab_mse = tf.reduce_mean(tf.square(tf.abs(Ab_pred_complex - Ab_ssfm)))
    return Ab_mse


def test_loss(batch, pinn_model):
    z_test, t_test, a_real, a_image = tf.split(batch, [1, 1, 1, 1], axis=-1)
    a_sffm = tf.cast(tf.complex(a_real, a_image), tf.complex64)

    a_pred = pinn_model(tf.concat([z_test, t_test], axis=-1))
    a_pred_real, a_pred_imag = tf.split(a_pred, 2, axis=-1)
    a_pred_complex = tf.complex(a_pred_real, a_pred_imag)

    testing_loss = tf.reduce_mean(tf.square(tf.abs(a_pred_complex - a_sffm)))
    return testing_loss


# Training step function
def train_step(model, optimizer, input_train_batch, A0_batch, boundary_batch, parameters, nlse_avg, a0_avg, ab_avg):
    with tf.GradientTape() as tape:
        loss = train_loss(input_train_batch, A0_batch, boundary_batch, model, parameters, nlse_avg, a0_avg, ab_avg)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

    # Function to plot pulse propagation using the trained model


def plot_model_pulse_propagation(model, standardized_input, standardization_params, parameters):
    # Constants
    T0 = parameters['T0']  # Initial pulse width (ps)
    L = parameters['L']  # Fiber length (km)
    dz = parameters['dz']  # Step size in z (km), reduced for higher accuracy
    T = parameters['T']  # Time window (ps)
    Nt = parameters['Nt']  # Increased number of time points for better resolution
    dt = parameters['dt']  # Step size in t (ps)
    beta_2 = parameters['beta_2']

    t = np.linspace(-T / 2, T / 2, Nt)
    L_D = T0 ** 2 / abs(beta_2)

    z = np.linspace(0, L, int(L / dz))

    # Get predictions from the model
    predictions = model.predict(standardized_input)

    input_mean, input_std, output_mean, output_std = standardization_params

    # De-normalize the predicted output
    predictions_real = predictions[:, 0] * output_std[0] + output_mean[0]
    predictions_imag = predictions[:, 1] * output_std[1] + output_mean[1]
    predictions_complex = predictions_real + 1j * predictions_imag

    A_t = predictions_complex.reshape(len(z), len(t))

    # Plot the results
    plt.figure(figsize=(20, 5))
    plt.imshow(np.abs(A_t).T, extent=[(z / L_D).min(), (z / L_D).max(), (t / T0).min(), (t / T0).max()], aspect='auto',
               origin='lower', cmap='jet')
    plt.colorbar(label='|A(z,t)|')
    plt.xlabel(f'Distance (z) | L_D = {round(L_D, 2)}Km')
    plt.ylabel(f'Time (t) | T0 = {T0}Ps')
    plt.title('Pulse propagation using trained model |A(z,t)|')
    plt.savefig('PINN pulse propagation')
    plt.close()

    # 3D plot
    Z, T = np.meshgrid(z / L_D, t / T0)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, T, np.abs(A_t.T), cmap='jet')
    ax.set_xlabel(f'Distance (z) | L_D = {round(L_D, 2)}Km')
    ax.set_ylabel(f'Time (t) | T0 = {T0}Ps')
    ax.set_zlabel('|A(z,t)|')
    ax.set_title('3D view of pulse propagation using trained model |A(z,t)|')
    plt.savefig('PINN pulse propagation 3D')
    plt.close()


def plot_ssfm_pulse_propagation(standardization_params, standardized_output_data, Z_grid, T_grid):
    input_mean, input_std, output_mean, output_std = standardization_params

    # De-normalize the standardized output data
    ssfm_real = standardized_output_data[:, 0] * output_std[0] + output_mean[0]
    ssfm_imag = standardized_output_data[:, 1] * output_std[1] + output_mean[1]
    ssfm_complex = ssfm_real + 1j * ssfm_imag

    # Reshape the predictions to match the grid shape
    A_ssfm = ssfm_complex.reshape(Z_grid.shape)

    # Calculate the extent values
    xmin, xmax = Z_grid.min(), Z_grid.max()
    ymin, ymax = T_grid.min(), T_grid.max()

    # Plot the results
    plt.figure(figsize=(20, 5))
    # Pass the calculated extent values
    plt.imshow(np.abs(A_ssfm).T, extent=[xmin, xmax, ymin, ymax],
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='|A(z,t)|')
    plt.xlabel(f'Distance (km)')
    plt.ylabel(f'Time (ps)')
    plt.title('SSFM - Pulse propagation |A(z,t)|')
    plt.savefig('SSFM pulse propagation')
    plt.close()


def plot_mse_heatmap(pinn_model, standardized_input, standardized_output, z_grid, t_grid, standardization_params):
    # Get predictions from the model
    predictions = pinn_model.predict(standardized_input)

    input_mean, input_std, output_mean, output_std = standardization_params

    # De-normalize the predicted output
    predictions_real = predictions[:, 0] * output_std[0] + output_mean[0]
    predictions_imag = predictions[:, 1] * output_std[1] + output_mean[1]
    predictions_complex = predictions_real + 1j * predictions_imag

    # De-normalize the SSFM output
    ssfm_real = standardized_output[:, 0] * output_std[0] + output_mean[0]
    ssfm_imag = standardized_output[:, 1] * output_std[1] + output_mean[1]
    ssfm_complex = ssfm_real + 1j * ssfm_imag

    # Determine the correct reshape dimensions
    reshape_dim1 = z_grid.shape[0]  # Number of z points
    reshape_dim2 = t_grid.shape[1]  # Number of t points

    # Reshape the predictions and SSFM output to match the grid shape
    predictions_complex = predictions_complex.reshape(reshape_dim1, reshape_dim2)
    ssfm_complex = ssfm_complex.reshape(reshape_dim1, reshape_dim2)

    # Calculate MSE
    mse = np.square(np.abs(predictions_complex - ssfm_complex))

    # Plot the MSE heatmap
    plt.figure(figsize=(20, 5))
    # Correct the extent parameter to include all four bounds
    plt.imshow(mse.T, extent=[z_grid.min(), z_grid.max(), t_grid.min(), t_grid.max()],
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='MSE')
    plt.xlabel('Distance (km)')
    plt.ylabel('Time (ps)')
    plt.title('MSE between A from PINN and A from SSFM')
    plt.savefig('MSE between A from PINN and A from SSFM')
    plt.close()