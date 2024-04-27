import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model


def build_pinn_model(input_shape=2, num_neurons=100, num_layers=4, output_shape=2):
    # Input layer: temporal-spatial coordinates (t, z)
    inputs = Input(shape=(input_shape,))

    # Hidden layers: using 'num_layers' dense layers with 'num_neurons' each and tanh activation
    x = Dense(num_neurons, activation='tanh')(inputs)
    for _ in range(num_layers - 1):
        x = Dense(num_neurons, activation='tanh')(x)

    # Output layer: real and imaginary parts of the complex envelope A
    outputs = Dense(output_shape, activation=None)(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model


# Define the physics-informed loss function within the main.py, so it has access to the pinn_model
def create_physics_informed_loss(pinn_model, beta_2, gamma, alpha):
    def nlse_loss(z, t, A_pred):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([z, t, A_pred])  # Ensure all variables are being watched

            # Calculate gradients outside the inner tape context
            # Note: Nested tape is required only if you need higher-order derivatives
            A_pred_z = tape.gradient(A_pred, z)
            A_pred_t = tape.gradient(A_pred, t)

        # Calculate second order derivative outside the first tape context
        with tf.GradientTape() as tape2:
            tape2.watch(t)
            # Recompute A_pred_t if necessary or ensure it's derived correctly
            A_pred_t = tape.gradient(A_pred, t)
            A_pred_tt = tape2.gradient(A_pred_t, t)

        # Now proceed with your loss calculation
        nlse_residual = A_pred_z + (1j * beta_2 / 2) * A_pred_tt - alpha * A_pred + 1j * gamma * tf.square(
            tf.abs(A_pred)) * A_pred
        nlse_loss_term = tf.reduce_mean(tf.square(tf.abs(nlse_residual)))

        return nlse_loss_term

    def composite_loss(y_true, A_pred):
        # Split the inputs and outputs
        A_true_real, A_true_imag, z, t = tf.split(y_true, [1, 1, 1, 1], axis=-1)
        A_pred_real, A_pred_imag = tf.split(A_pred, 2, axis=-1)

        # Convert to complex number
        A_true = tf.complex(A_true_real, A_true_imag)
        A_pred_complex = tf.complex(A_pred_real, A_pred_imag)

        # Compute the prediction error
        prediction_error = tf.reduce_mean(tf.square(tf.abs(A_pred_complex - A_true)))

        # Compute the NLSE loss
        physics_loss = nlse_loss(z, t, A_pred_complex)

        # Combine the prediction error with the physics-informed loss
        total_loss = prediction_error + physics_loss

        return total_loss

    # Return the composite loss function
    return composite_loss


def ssfm(A0, dz, dz_steps, dt, t_steps, beta_2, gamma, alpha):
    # Discretize time and space
    z = np.arange(0, dz * dz_steps, dz)
    t = np.arange(-t_steps * dt / 2, t_steps * dt / 2, dt)

    # Pre-compute the linear operator
    omega = 2 * np.pi * np.fft.fftfreq(t.size, dt)
    linear_operator = np.exp(-0.5 * (1j * beta_2 * omega ** 2 - alpha) * dz)

    # Initialize the field
    A = np.zeros((dz_steps, t.size), dtype=complex)
    A[0, :] = A0(t)

    # SSFM loop
    for i in range(1, dz_steps):
        # Linear step
        A[i, :] = np.fft.ifft(np.fft.fft(A[i - 1, :]) * linear_operator)

        # Nonlinear step
        A[i, :] = A[i, :] * np.exp(1j * gamma * np.abs(A[i, :]) ** 2 * dz)

    return z, t, A


# Define the initial pulse shape, e.g., a Gaussian pulse

def gaussian_pulse(T, pulse_width=1.0, peak_power=1.0):
    return np.sqrt(peak_power) * np.exp(-T**2 / (2 * pulse_width**2))


def generate_training_data(A0, fiber_length, num_steps, dt, dz, beta_2, gamma, alpha):
    T = np.arange(-num_steps // 2, num_steps // 2) * dt
    Z = np.arange(0, fiber_length, dz)
    W = np.fft.fftfreq(T.size, d=dt) * 2 * np.pi
    W = np.fft.fftshift(W)  # Shift zero frequency to center

    # Convert alpha from dB/km to 1/m if necessary
    # alpha_m = alpha / (10 * 1e3 * np.log(10))

    # Initial pulse in the time domain
    A = np.zeros((len(Z), len(T)), dtype=complex)
    A[0, :] = A0(T)

    # SSFM loop
    for i in range(1, len(Z)):
        # Apply half the attenuation for the step
        A[i - 1, :] = A[i - 1, :] * np.exp(-alpha * dz / 2)

        # Linear step in the frequency domain
        A_fft = np.fft.fft(A[i - 1, :])
        A_fft = A_fft * np.exp(-1j * (beta_2 / 2) * W ** 2 * dz)

        # Nonlinear step in the time domain
        A[i, :] = np.fft.ifft(A_fft)
        A[i, :] = A[i, :] * np.exp(1j * gamma * np.abs(A[i, :]) ** 2 * dz)

        # Apply the second half of the attenuation for the step
        A[i, :] = A[i, :] * np.exp(-alpha * dz / 2)

    return Z, T, A


# Assuming 'A', 'Z', T are your outputs from the SSFM function
def plot_results(Z, T, A):
    # Selecting a specific output position for detailed plot, here at the end of the fiber
    A_output = A[-1, :]

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot the intensity of the pulse at the end of the fiber
    axs[0, 0].plot(T, np.abs(A_output)**2)
    axs[0, 0].set_title('Output Pulse Intensity')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Intensity |A|^2')

    # Plot the phase of the pulse at the end of the fiber
    axs[0, 1].plot(T, np.angle(A_output))
    axs[0, 1].set_title('Output Pulse Phase')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Phase (radians)')

    # Intensity plot along the fiber
    intensity_along_fiber = np.abs(A)**2
    contour = axs[1, 0].contourf(T, Z, intensity_along_fiber, levels=100, cmap='hot')
    fig.colorbar(contour, ax=axs[1, 0])
    axs[1, 0].set_title('Intensity Along the Fiber')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Propagation Distance (m)')

    # Plot the peak power evolution along the fiber
    peak_power = np.max(intensity_along_fiber, axis=1)
    axs[1, 1].plot(Z, peak_power)
    axs[1, 1].set_title('Peak Power Evolution Along the Fiber')
    axs[1, 1].set_xlabel('Propagation Distance (m)')
    axs[1, 1].set_ylabel('Peak Power')

    plt.tight_layout()
    plt.show()


def standardize_data(input_data, output_data):
    # Calculate mean and standard deviation for input data
    input_mean = input_data.mean(axis=0)
    input_std = input_data.std(axis=0)

    # Avoid division by zero in case of a constant feature
    input_std[input_std == 0] = 1

    # Standardize input data
    standardized_input = (input_data - input_mean) / input_std

    # Calculate mean and standard deviation for output data
    output_mean = output_data.mean(axis=0)
    output_std = output_data.std(axis=0)

    # Avoid division by zero in case of a constant feature
    output_std[output_std == 0] = 1

    # Standardize output data
    standardized_output = (output_data - output_mean) / output_std

    # Return the standardized data and the parameters used for standardization
    return standardized_input, standardized_output, (input_mean, input_std, output_mean, output_std)

