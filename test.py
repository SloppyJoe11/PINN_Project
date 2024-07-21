import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft, fftshift

# Constants
T0 = 20e-12  # Pulse width (s)
A0 = 1  # Pulse amplitude
L = 80e3  # Fiber length (m)
alpha = 0  # Attenuation coefficient in m^-1
beta2 = -20e-27  # GVD parameter (s^2/m)  # TODO: find out why -27 doesnt work well
gamma = 0 # Non-linearity parameter (1/(W*m))
dz = 500  # Step size in z (m), reduced for higher accuracy

# Time grid
T = 40 * T0
Nt = 512  # Increased number of time points for better resolution
dt = T / Nt  # Should be at least T0/10
t = np.linspace(-T / 2, T / 2, Nt)

# Initial pulse - gaussian
A = A0 * np.exp(-t ** 2 / T0 ** 2)

# Frequency grid
f = np.fft.fftfreq(Nt, dt)
omega = 2 * np.pi * f

# Calculate dispersion length
L_D = T0**2 / abs(beta2)   # TODO: after fixing e-28, delete *10


# Generate the training data
# Propagation
def ssfm(A, dz, L, beta2, gamma, omega, alpha):
    Nz = int(L / dz)
    A_t = np.zeros((Nz, Nt), dtype=np.complex_)
    A_t[0, :] = A

    for i in range(1, Nz):
        # Nonlinear step
        nonlinear_phase = np.exp(1j * gamma * np.abs(A_t[i - 1, :]) ** 2 * dz / 2)
        A_tilde = A_t[i - 1, :] * nonlinear_phase

        # Linear step
        A_f = fft(A_tilde)
        linear_phase = np.exp(-0.5j * beta2 * omega ** 2 * dz)
        A_f = A_f * linear_phase
        A_tilde = ifft(A_f)

        # Nonlinear step again (completing the half-step)
        nonlinear_phase = np.exp(1j * gamma * np.abs(A_tilde) ** 2 * dz / 2)
        A_t[i, :] = A_tilde * nonlinear_phase

        # Attenuation step
        A_t[i, :] = A_t[i, :] * np.exp(-alpha * dz)

    return A_t


# Normalize data
def standardize_data(input_data, output_data):
    input_mean = input_data.mean(axis=0)
    input_std = input_data.std(axis=0)
    input_std[input_std == 0] = 1
    standardized_input = (input_data - input_mean) / input_std

    output_mean = output_data.mean(axis=0)
    output_std = output_data.std(axis=0)
    output_std[output_std == 0] = 1
    standardized_output = (output_data - output_mean) / output_std

    return standardized_input, standardized_output, (input_mean, input_std, output_mean, output_std)


def plot_pulse_3d(z, t, A_t, T0, L_D, Nt):
    Z, T = np.meshgrid(z, t)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, T, np.abs(A_t.T), cmap='jet')
    ax.set_xlabel(f'Distance (m) | L_D = {L_D/1e3}Km')
    ax.set_ylabel(f'Time (s) | T0 = {T0/1e-12}Ps')
    ax.set_zlabel('|A(z,t)|')
    ax.set_title(f'3D view of SSFM - Pulse propagation |A(z,t)| {Nt} time samples')
    ax.set_zlim(0, 1)  # Lower the amplitude range
    plt.show()


def plot_pulse_2d(z, t, A_t, T0, L_D, Nt):
    plt.figure(figsize=(20, 5))
    plt.imshow(np.abs(A_t).T, extent=[(z/L_D).min(), (z/L_D).max(), (t/T0).min(), (t/T0).max()], aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='|A(z,t)|')
    plt.xlabel(f'Distance (m) | L_D = {L_D/1e3}Km')
    plt.ylabel(f'Time (s) | T0 = {T0/1e-12}Ps')
    plt.title(f'SSFM - Pulse propagation |A(z,t)| {Nt} time samples')
    plt.show()


# Generate the training data
A = ssfm(A, dz, L, beta2, gamma, omega, alpha)
Z = np.linspace(0, L, int(L / dz))
T_ = np.linspace(-T / 2, T / 2, Nt)


# Plot results
plot_pulse_2d(Z, T_, A, T0, L_D, Nt)