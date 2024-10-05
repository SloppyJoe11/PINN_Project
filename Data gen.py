import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from scipy.fft import fft, ifft, fftshift

# Constants
alpha = 0  # Attenuation coefficient in m^-1
beta2 = -20  # GVD parameter ((ps)^2/km)
gamma = 1.27  # Non-linearity parameter (1/(W*km))

P0 = 1e-3  # Pulse Power (W)
A0 = np.sqrt(P0)  # Pulse Amplitude (W)
T0 = 20  # Pulse width (ps)

# Calculate dispersion length
L_D = T0**2 / abs(beta2)
L_n = 1 / (gamma*P0)

L = 4*L_D  # Fiber length (km)
Nz = 800
dz = L / Nz  # Step size in z (km), reduced for higher accuracy

# Splitting
train_percentage = 0.725  # Train ratio out of total number of data samples
validation_percentage = 0.025  # Validation ratio out of total number of data samples

# Time grid
T = 40 * T0
Nt = 512  # Increased number of time points for better resolution
dt = T / Nt
t = np.linspace(-T / 2, T / 2, Nt)


# Initial pulse - gaussian
A = A0 * np.exp(-t ** 2 / T0 ** 2)

# Initial pulse - Multiple gaussian
# A = A0 * (np.exp(-t ** 2 / T0 ** 2) + 0.75*np.exp(-(t+10*T0) ** 2 / T0 ** 2) + 0.5*np.exp(-(t-10*T0) ** 2 / T0 ** 2)
#          + 0.25*np.exp(-(t+5*T0) ** 2 / T0 ** 2) + np.exp(-(t-5*T0) ** 2 / T0 ** 2))

# Initial pulse - Soliton
# A = A0 / np.cosh(t/T0)

# Frequency grid
f = np.fft.fftfreq(Nt, dt)
omega = 2 * np.pi * f

print(f'L_n = {L_n}, L_D = {L_D}')


# Generate the training data
# Propagation
def ssfm(A, dz, L, beta2, gamma, omega, alpha):
    Nz = int(L / dz)
    A_t = np.zeros((Nz, Nt), dtype=np.complex64)
    A_t[0, :] = A

    for i in range(1, Nz):

        # Nonlinear half step
        nonlinear_phase = np.exp(1j * gamma * np.abs(A_t[i-1, :]) ** 2 * dz / 2)
        A_tilde = A_t[i-1, :] * nonlinear_phase

        # Linear step
        A_f = fft(A_tilde)
        linear_phase = np.exp(-0.5j * beta2 * omega ** 2 * dz)
        A_f = A_f * linear_phase
        A_tilde = ifft(A_f)

        # Nonlinear half step
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
    Z, T = np.meshgrid(z / L_D, t / T0)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, T, np.abs(A_t.T), cmap='jet')
    ax.set_xlabel(f'Distance (L_D) | L_D = {round(L_D, 2)}Km')
    ax.set_ylabel(f'Time (T0) | T0 = {T0}Ps')
    ax.set_zlabel('|A(z,t)|')
    ax.set_title(f'3D view of SSFM - Pulse propagation |A(z,t)| {Nt} time samples, '
                 f'alpha = {alpha}, beta = {beta2}, gamma = {gamma}')
    plt.savefig('SSFM pulse 3d.png')
    plt.close()


def plot_pulse_2d(z, t, A_t, T0, L_D, Nt):
    plt.figure(figsize=(20, 5))
    plt.imshow(np.abs(A_t).T, extent=[(z/L_D).min(), (z/L_D).max(), (t/T0).min(), (t/T0).max()], aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='|A(z,t)|')
    plt.xlabel(f'Distance (L_D) - L_D = {round(L_D, 2)}Km')
    plt.ylabel(f'Time (T0) - T0 = {T0}Ps')
    plt.title(f'SSFM - Pulse propagation |A(z,t)| {Nt} time samples, '
              f'alpha = {alpha}, beta = {beta2}, gamma = {gamma}')
    plt.savefig('SSFM pulse 2d.png')
    plt.close()


def plot_initial_final_pulse(A_t, t):
    plt.figure(figsize=(20, 5))
    plt.plot(t, np.abs(A_t[0, :]), label='Initial Pulse (A0)')
    plt.plot(t, np.abs(A_t[-1, :]), label='Final Pulse (A final)')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude')
    plt.title('Initial and Final Pulse Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('Initial and Final Pulse Comparison')
    plt.close()


# Generate the training data
A = ssfm(A, dz, L, beta2, gamma, omega, alpha)
Z = np.linspace(0, L, int(L / dz))
T_ = np.linspace(-T / 2, T / 2, Nt)

# A_normalized = np.abs(A) / np.max(np.abs(A))

# Plot results
plot_pulse_2d(Z, T_, A, T0, L_D, Nt)
plot_pulse_3d(Z, T_, A, T0, L_D, Nt)
plot_initial_final_pulse(A, t)

# Create a 2D grid of Z and T values
Z_grid, T_grid = np.meshgrid(Z, T_, indexing='ij')
input_data = np.vstack((Z_grid.flatten(), T_grid.flatten())).T

# Flatten A to have the same shape as input_data
output_data = A.flatten()
output_data = np.stack((output_data.real, output_data.imag), axis=-1)

combined_data = np.concatenate((input_data, output_data), axis=-1)

z_values = combined_data[:, 0]
train_split_index = np.searchsorted(z_values, train_percentage * 4*L_D)
val_split_index = np.searchsorted(z_values, (train_percentage + validation_percentage) * 4*L_D)

# Normalize the input and output data
standardized_input_data, standardized_output_data, standardization_params = standardize_data(input_data, output_data)
standardized_combined_data = np.concatenate((standardized_input_data, standardized_output_data), axis=-1)

# Split data 0-2.4L_D (60%) for train, 2.4L_D-3L_D (15%) for validation and 3L_D-4L_D (25%) for test
train_data = standardized_combined_data[:train_split_index]
val_data = standardized_combined_data[train_split_index: val_split_index]
test_data = standardized_combined_data[val_split_index:]


# Dictionary of parameters
parameters = {
    'T0': T0,
    'P0': P0,
    'A0': A0,
    'L': L,
    'L_D': L_D,
    'L_n': L_n,
    'alpha': alpha,
    'beta2': beta2,
    'gamma': gamma,
    'dz': dz,
    'T': T,
    'Nt': Nt,
    'dt': dt,
    't': t,
    'train_percentage': train_percentage,
    'validation_percentage': validation_percentage,
}

with open('parameters.pkl', 'wb') as f:
    pickle.dump(parameters, f)

print("Parameters saved to 'parameters.pkl'")

# Save the processed data
np.savez('processed_training_data.npz',

         train_data=train_data,
         test_data=test_data,                   val_data=val_data,

         Z_grid=Z_grid,                         T_grid=T_grid,

         standardized_combined_data=standardized_combined_data,
         standardized_input_data=standardized_input_data,
         standardized_output_data=standardized_output_data,
         standardization_params=standardization_params
         )

print("Processed training data saved to 'processed_training_data.npz'")
