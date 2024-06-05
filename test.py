import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the NLSE
beta_2 = -21.27e-27  # s^2/m
gamma = 1.3e-3  # 1/(W*m)
alpha = 0.046 / 1000  # 1/m (dB/km converted to 1/m)

# Parameters for data generation
fiber_length = 10000  # meters
num_steps = 1024
dt = 1e-3  # seconds
dz = 1  # meters


def generate_training_data(A0, fiber_length, num_steps, dt, dz, beta_2, gamma, alpha):
    T = np.arange(-num_steps // 2, num_steps // 2) * dt
    Z = np.arange(0, fiber_length, dz)
    W = np.fft.fftfreq(T.size, d=dt) * 2 * np.pi
    W = np.fft.fftshift(W)
    A = np.zeros((len(Z), len(T)), dtype=complex)
    A[0, :] = A0(T)

    for i in range(1, len(Z)):
        A[i - 1, :] = A[i - 1, :] * np.exp(-alpha * dz / 2)
        A_fft = np.fft.fft(A[i - 1, :])
        A_fft = A_fft * np.exp(-1j * (beta_2 / 2) * W ** 2 * dz)
        A[i, :] = np.fft.ifft(A_fft)
        A[i, :] = A[i, :] * np.exp(1j * gamma * np.abs(A[i, :]) ** 2 * dz)
        A[i, :] = A[i, :] * np.exp(-alpha * dz / 2)

    return Z, T, A


# Define the initial pulse shape, e.g., a Gaussian pulse
def gaussian_pulse(T, pulse_width=0.05, peak_power=1.0):
    return np.sqrt(peak_power) * np.exp(-T ** 2 / (2 * pulse_width ** 2))


# 3D Plotting the result
def plot_3d(z, t, a):
    Z, T = np.meshgrid(z, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, Z, np.abs(a.T), cmap='viridis')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Amplitude')
    ax.set_title('Pulse Propagation in Fiber Optic')
    plt.show()


if __name__ == "__main__":
    Z, T, A = generate_training_data(gaussian_pulse, fiber_length, num_steps, dt, dz, beta_2, gamma, alpha)

    # 2D Plotting the result
    plt.imshow(np.abs(A) ** 2, extent=[T[0], T[-1], Z[-1], Z[0]], aspect='auto', cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.title('Pulse Propagation in Fiber Optic')
    plt.colorbar(label='|A|^2')
    plt.show()

    # 3D Plotting the result
    plot_3d(Z, T, A)
