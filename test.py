import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft, fftshift

# Constants
c = 3e8  # speed of light in vacuum (m/s)
T0 = 20e-12  # Pulse width (s)
A0 = 1  # Pulse amplitude
L = 80e3  # Fiber length (m)
beta2 = -20e-27  # GVD parameter (s^2/m)
gamma = 1.27e-3  # Nonlinearity parameter (1/(W*m))
dz = 0.1e3  # Step size in z (m), reduced for higher accuracy

# Time grid
T = 40 * T0
Nt = 1024  # Increased number of time points for better resolution
dt = T / Nt
t = np.linspace(-T / 2, T / 2, Nt)

# Initial pulse - gaussian
A = A0 * np.exp(-t ** 2 / T0 ** 2)

# Frequency grid
f = np.fft.fftfreq(Nt, dt)
omega = 2 * np.pi * f

# Calculate dispersion length
L_D = T0**2 / abs(beta2)


# Propagation
def ssfm(A, dz, L, beta2, gamma, omega):
    Nz = int(L / dz)
    A_t = np.zeros((Nz, Nt), dtype=np.complex_)
    A_t[0, :] = A

    for i in range(1, Nz):
        # Linear step
        A_f = fft(A_t[i - 1, :])
        A_f = A_f * np.exp(-0.5j * beta2/10 * omega ** 2 * dz) # # TODO: find out why beta2/10, needs to be beta2 only
        A_tilde = ifft(A_f)

        # Nonlinear step
        A_t[i, :] = A_tilde * np.exp(1j * gamma * np.abs(A_tilde) ** 2 * dz)

    return A_t


# Run SSFM
A_t = ssfm(A, dz, L, beta2, gamma, omega)

# Plotting functions


def plot_pulse_2d(z, t, A_t, T0, L_D, Nt):
    plt.figure(figsize=(20, 5))
    plt.imshow(np.abs(A_t).T, extent=[(z/L_D).min(), (z/L_D).max(), (t/T0).min(), (t/T0).max()], aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='|A(z,t)|')
    plt.xlabel(f'Distance (z/L_D) - L_D = {round(L_D/1e3, 2)}Km')
    plt.ylabel(f'Time (t/T0) - T0 = {T0/1e-12}Ps')
    plt.title(f'SSFM - Pulse propagation |A(z,t)| {Nt} time samples')
    plt.show()


def plot_pulse_3d(z, t, A_t, T0, L_D, Nt):
    Z, T = np.meshgrid(z / L_D, t / T0)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, T, np.abs(A_t.T), cmap='jet')
    ax.set_xlabel(f'Distance (z/L_D) - L_D = {round(L_D/1e3, 2)}Km')
    ax.set_ylabel(f'Time (t/T0) - T0 = {T0/1e-12}Ps')
    ax.set_zlabel('|A(z,t)|')
    ax.set_title(f'3D view of SSFM - Pulse propagation |A(z,t)| {Nt} time samples')
    ax.set_zlim(0, 1)  # Lower the amplitude range
    plt.show()


# Create z grid
Nz = int(L / dz)
z = np.linspace(0, L, Nz)


# Plot results
plot_pulse_2d(z, t, A_t, T0, L_D, Nt)
plot_pulse_3d(z, t, A_t, T0, L_D, Nt)
