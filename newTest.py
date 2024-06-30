import numpy as np
import matplotlib.pyplot as plt

# Parameters for the NLSE
beta_2 = -20 * 1e-27  # s^2/m (converted from ps^2/km to s^2/m)
gamma = 1.3 * 1e-3  # (V*m)^-1
alpha = 0  # m^-1, assuming no attenuation for simplicity

# Function to generate initial pulse
def A0(T, pulse_width):
    return np.exp(-T**2 / (2 * pulse_width**2))


def plot_initial_pulse(T, A0):
    plt.figure(figsize=(10, 6))
    plt.plot(T, np.abs(A0), label='Initial Pulse |A(0, T)|')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude')
    plt.title('Initial Pulse Shape')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pulse(A,T,Z):
    # Plot results (example)
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(A), extent=[T.min() * 1e12, T.max() * 1e12, Z.max(), Z.min()], aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (ps)')
    plt.ylabel('Distance (m)')
    plt.title('Pulse Evolution in Fiber (SSFM)')
    plt.show()


# Function to generate training data using SSFM
def generate_ssfm_data(fiber_length_km, pulse_width_ps, dz_m, beta_2, gamma, alpha):
    # Convert parameters
    fiber_length_m = fiber_length_km * 1e3  # convert km to m
    pulse_width = pulse_width_ps  # pulse width in ps
    dz = dz_m  # step size in m

    # Time and frequency grids
    num_steps = 64  # Number of time samples for higher resolution
    dt = pulse_width_ps / 10  # Time step in ps
    T = np.arange(-num_steps // 2, num_steps // 2) * dt  # Time vector in s
    W = np.fft.fftfreq(T.size, d=dt) * 2 * np.pi  # Frequency vector in rad/s
    W = np.fft.fftshift(W)  # Shift zero frequency component to the center

    # Spatial grid
    Z = np.arange(0, fiber_length_m, dz)

    # Initialize the pulse
    A = np.zeros((len(Z), len(T)), dtype=complex)
    A[0, :] = A0(T, pulse_width)

    # Propagate the pulse using SSFM
    for i in range(1, len(Z)):
        A[i - 1, :] *= np.exp(-alpha * dz / 2)
        A_fft = np.fft.fft(A[i - 1, :])
        A_fft *= np.exp(-1j * (beta_2 / 2) * W ** 2 * dz)
        A[i, :] = np.fft.ifft(A_fft)
        A[i, :] *= np.exp(1j * gamma * np.abs(A[i, :]) ** 2 * dz)
        A[i, :] *= np.exp(-alpha * dz / 2)

    return Z, T, A

# Define parameters
fiber_length_km = 500  # Fiber length in km
pulse_width_ps = 30 * 1e-12  # Pulse width in ps
dz_m = 100  # Step size in m

# Generate data
Z, T, A = generate_ssfm_data(fiber_length_km, pulse_width_ps, dz_m, beta_2, gamma, alpha)

plot_initial_pulse(T, A[0, :])
plot_pulse(A, T, Z)

