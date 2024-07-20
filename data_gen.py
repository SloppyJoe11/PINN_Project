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
gamma = 1.27e-3  # Non-linearity parameter (1/(W*m))
dz = 0.1e3  # Step size in z (m), reduced for higher accuracy

# Time grid
T = 40 * T0
Nt = 512  # Increased number of time points for better resolution
dt = T / Nt
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
        # Linear step
        A_f = fft(A_t[i - 1, :])
        A_f = A_f * np.exp(-0.5j * beta2 * omega ** 2 * dz)
        A_tilde = ifft(A_f)

        # Nonlinear step
        A_tilde = A_tilde * np.exp(1j * gamma * np.abs(A_tilde) ** 2 * dz)

        # Attenuation step
        A_t[i, :] = A_tilde * np.exp(-alpha * dz)

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
    ax.set_xlabel(f'Distance (z/L_D) - L_D = {round(L_D/1e3, 2)}Km')
    ax.set_ylabel(f'Time (t/T0) - T0 = {T0/1e-12}Ps')
    ax.set_zlabel('|A(z,t)|')
    ax.set_title(f'3D view of SSFM - Pulse propagation |A(z,t)| {Nt} time samples')
    ax.set_zlim(0, 1)  # Lower the amplitude range
    plt.show()


def plot_pulse_2d(z, t, A_t, T0, L_D, Nt):
    plt.figure(figsize=(20, 5))
    plt.imshow(np.abs(A_t).T, extent=[(z/L_D).min(), (z/L_D).max(), (t/T0).min(), (t/T0).max()], aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='|A(z,t)|')
    plt.xlabel(f'Distance (z/L_D) - L_D = {round(L_D/1e3, 2)}Km')
    plt.ylabel(f'Time (t/T0) - T0 = {T0/1e-12}Ps')
    plt.title(f'SSFM - Pulse propagation |A(z,t)| {Nt} time samples')
    plt.show()


# Generate the training data
A = ssfm(A, dz, L, beta2, gamma, omega, alpha)
Z = np.linspace(0, L, int(L / dz))
T_ = np.linspace(-T / 2, T / 2, Nt)


# Plot results
plot_pulse_2d(Z, T_, A, T0, L_D, Nt)
plot_pulse_3d(Z, T_, A, T0, L_D, Nt)


# Create a 2D grid of Z and T values
Z_grid, T_grid = np.meshgrid(Z, T_, indexing='ij')
input_data = np.vstack((Z_grid.flatten(), T_grid.flatten())).T

# Flatten A to have the same shape as input_data
output_data = A.flatten()
output_data = np.stack((output_data.real, output_data.imag), axis=-1)

combined_data = np.concatenate((input_data, output_data), axis=-1)

# Calculate indices for A_0 and boundary conditions on original combined data
A_0_indices = np.where(combined_data[:, 0] == 0)[0]

boundary_indices_minus_T = np.where(combined_data[:, 1] == -T/2)[0]
boundary_indices_plus_T = np.where(combined_data[:, 1] == T/2)[0]

# Normalize the input and output data
standardized_input_data, standardized_output_data, standardization_params = standardize_data(input_data, output_data)
standardized_combined_data = np.concatenate((standardized_input_data, standardized_output_data), axis=-1)

# Split the dataset into training and (validation + test)
input_train, input_val_test, output_train, output_val_test = train_test_split(
    standardized_input_data,
    standardized_output_data,
    test_size=0.3,
    random_state=42)

# Further split for validation and test sets
input_val, input_test, output_val, output_test = train_test_split(
    input_val_test,
    output_val_test,
    test_size=0.5,
    random_state=42)

# Extract the rows for A_0 and boundary conditions
A_0 = standardized_combined_data[A_0_indices]
boundary_A_minus_T = standardized_combined_data[boundary_indices_minus_T]
boundary_A_plus_T = standardized_combined_data[boundary_indices_plus_T]
A_boundary = np.concatenate((boundary_A_minus_T, boundary_A_plus_T), axis=0)

# Split A_0 and A_boundary into training and validation sets
A0_train, A0_val = train_test_split(A_0, test_size=0.3, random_state=42)
A_boundary_train, A_boundary_val = train_test_split(A_boundary, test_size=0.3, random_state=42)

# Save the processed data
np.savez('processed_training_data.npz',

         input_train=input_train,               output_train=output_train,
         input_val=input_val,                   output_val=output_val,
         input_test=input_test,                 output_test=output_test,
         A0_train=A0_train,                     A0_val=A0_val,
         A_boundary_train=A_boundary_train,     A_boundary_val=A_boundary_val,
         standardized_input_data=standardized_input_data,
         Z_grid=Z_grid,                         T_grid=T_grid,
         standardization_params=standardization_params)

print("Processed training data saved to 'processed_training_data.npz'")


