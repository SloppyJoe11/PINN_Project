import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift

# Parameters
beta2 = -20e-27  # s^2/m
gamma = 1.3e-3  # 1/(W*m)
T0 = 1e-12  # s
P0 = 1  # W (Peak power)
C = 0  # Chirp parameter
Ld = T0**2/abs(beta2)

# Simulation parameters
Nt = 512  # Number of time samples
Nz = 10000  # Number of distance steps
time_window = 40 * T0
z_final = 4 * Ld  # 5 times the dispersion length

# Create arrays
t = np.linspace(-time_window / 2, time_window / 2, Nt)
z = np.linspace(0, z_final, Nz)
dt = t[1] - t[0]
dz = z[1] - z[0]

# Frequency array
omega = 2 * np.pi * fftshift(np.fft.fftfreq(Nt, dt))

# Initial pulse (Gaussian)
A0 = np.sqrt(P0) * np.exp(-(1 + 1j * C) * (t / (2 * T0)) ** 2)

# Create grid for A(z,t)
A = np.zeros((Nz, Nt), dtype=complex)
A[0, :] = A0

# SSFM
for i in range(1, Nz):
    # Nonlinear step
    A_nl = A[i - 1, :] * np.exp(1j * gamma * dz * np.abs(A[i - 1, :]) ** 2)

    # Linear step (in frequency domain)
    A_lin = ifft(np.exp(-1j * beta2 / 2 * omega ** 2 * dz) * fft(A_nl))

    A[i, :] = A_lin

# Plot results
plt.figure(figsize=(12, 6))
plt.imshow(np.abs(A) ** 2, aspect='auto', extent=[t[0] * 1e12, t[-1] * 1e12, z[-1] / 1000, z[0] / 1000])
plt.colorbar(label='|A|^2 (W)')
plt.xlabel('Time (ps)')
plt.ylabel('Distance (km)')
plt.title('Pulse Propagation')
plt.show()

# Plot initial and final pulses
plt.figure(figsize=(10, 6))
plt.plot(t * 1e12, np.abs(A[0, :]) ** 2, label='Initial')
plt.plot(t * 1e12, np.abs(A[-1, :]) ** 2, label='Final')
plt.xlabel('Time (ps)')
plt.ylabel('Power (W)')
plt.legend()
plt.title('Initial and Final Pulse Shapes')
plt.show()