import numpy as np
from sklearn.model_selection import train_test_split

# Parameters for the NLSE
beta_2 = -21.27e-27  # s^2/m
gamma = 1.3e-3  # 1/(W*m)
alpha = 0.046 / 1000  # Convert from dB/km if needed, else use direct 1/m


# Generate the training data
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
def gaussian_pulse(T, pulse_width=1.0, peak_power=1.0):
    return np.sqrt(peak_power) * np.exp(-T**2 / (2 * pulse_width**2))


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


# Parameters for data generation
fiber_length = 100  # meters
num_steps = 1024
dt = 1e-4  # seconds
dz = 0.1  # meters

# Generate the training data
Z, T, A = generate_training_data(gaussian_pulse, fiber_length, num_steps, dt, dz, beta_2, gamma, alpha)

# Create a 2D grid of Z and T values
Z_grid, T_grid = np.meshgrid(Z, T, indexing='ij')
input_data = np.vstack((Z_grid.flatten(), T_grid.flatten())).T

# Flatten A to have the same shape as input_data
output_data = A.flatten()
output_data = np.stack((output_data.real, output_data.imag), axis=-1)

# Normalize the input and output data
standardized_input_data, standardized_output_data, standardization_params = standardize_data(input_data, output_data)
standardized_output_data = np.concatenate((standardized_output_data, input_data), axis=-1)

# Split the dataset into training and (validation + test)
input_train, input_val_test, output_train, output_val_test = train_test_split(
    standardized_input_data,
    standardized_output_data,
    test_size=0.3,
    random_state=42
)

# Further split for validation and test sets
input_val, input_test, output_val, output_test = train_test_split(
    input_val_test,
    output_val_test,
    test_size=0.5,
    random_state=42
)

# Save the processed data
np.savez('processed_training_data.npz',
         input_train=input_train, output_train=output_train,
         input_val=input_val, output_val=output_val,
         input_test=input_test, output_test=output_test)
print("Processed training data saved to 'processed_training_data.npz'")
