import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Parameters for the NLSE
beta_2 = -21.27e-27  # s^2/m
gamma = 1.3e-3  # 1/(W*m)
alpha = 0.046 / 1000  # Convert from dB/km if needed, else use direct 1/m


# Build the model
def build_pinn_model(input_shape=2, num_neurons=100, num_layers=4, output_shape=2):
    inputs = Input(shape=(input_shape,))
    x = Dense(num_neurons, activation='tanh')(inputs)
    for _ in range(num_layers - 1):
        x = Dense(num_neurons, activation='tanh')(x)
    outputs = Dense(output_shape, activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


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

# Plot training history
def plot_history(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

# Combined loss function
def combined_loss(y_true, y_pred):
    A_true_real, A_true_imag, z, t = tf.split(y_true, [1, 1, 1, 1], axis=-1)
    A_pred_real, A_pred_imag = tf.split(y_pred, 2, axis=-1)

    # Combine real and imaginary parts to form complex numbers
    A_true = tf.complex(tf.cast(A_true_real, dtype=tf.float32), tf.cast(A_true_imag, dtype=tf.float32))
    A_pred = tf.complex(tf.cast(A_pred_real, dtype=tf.float32), tf.cast(A_pred_imag, dtype=tf.float32))

    # Prediction error (MSE)
    prediction_error = tf.reduce_mean(tf.square(tf.abs(A_pred - A_true)))

    # NLSE residual
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([z, t])
        A_pred = pinn_model(tf.concat([z, t], axis=1))
        A_pred_z = tape.gradient(A_pred, z)
        A_pred_t = tape.gradient(A_pred, t)

    A_pred_tt = tape.gradient(A_pred_t, t)

    del tape

    # Ensure all relevant variables are complex64
    A_pred = tf.cast(A_pred, dtype=tf.complex64)
    abs_A_pred_squared = tf.square(tf.abs(A_pred))  # This remains real
    A_pred_z = tf.cast(A_pred_z, dtype=tf.complex64)
    A_pred_tt = tf.cast(A_pred_tt, dtype=tf.complex64)
    beta_2_complex = tf.cast(beta_2, dtype=tf.complex64)
    alpha_complex = tf.cast(alpha, dtype=tf.complex64)
    gamma_complex = tf.cast(gamma, dtype=tf.complex64)

    # Compute NLSE residual terms
    attenuation_complex = alpha_complex * A_pred
    chrom_dis_complex = (1j * beta_2_complex / 2) * A_pred_tt
    non_lin_complex = 1j * gamma_complex * tf.cast(abs_A_pred_squared, dtype=tf.complex64) * A_pred

    # NLSE residual calculation
    nlse_residual = A_pred_z + chrom_dis_complex - attenuation_complex + non_lin_complex
    nlse_loss_term = tf.reduce_mean(tf.square(tf.abs(nlse_residual)))

    # Combine NLSE loss and prediction error
    total_loss = tf.add(prediction_error, nlse_loss_term)

    return total_loss


# Training step function
def train_step(model, optimizer, loss_fn, x_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(x_batch, training=True)
        loss = loss_fn(y_batch, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Main code
pinn_model = build_pinn_model()

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

# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# Training parameters
epochs = 20
batch_size = 1024

input_train = tf.cast(input_train, tf.float32)
output_train = tf.cast(output_train, tf.float32)
input_val = tf.cast(input_val, tf.float32)

# Prepare the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((input_train, output_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((input_val, output_val)).batch(batch_size)

# Custom training loop
history = {'loss': [], 'val_loss': []}

print(f"Train dataset size: {len(list(train_dataset))} batches")
print(f"Validation dataset size: {len(list(val_dataset))} batches")

for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()

    batch_num = 1

    # Training loop
    for x_batch, y_batch in train_dataset:
        loss = train_step(pinn_model, optimizer, combined_loss, x_batch, y_batch)
        epoch_loss_avg.update_state(loss)
        batch_num += 1
        if (batch_num % 100) == 0:
            print(f"Training batch number {batch_num}, loss: {loss:.4f}")

    # Validation loop
    for x_batch_val, y_batch_val in val_dataset:
        y_pred_val = pinn_model(x_batch_val, training=False)
        val_loss = combined_loss(y_batch_val, y_pred_val)
        epoch_val_loss_avg.update_state(val_loss)
        batch_num += 1
        if (batch_num % 100) == 0:
            print(f"Validation batch number {batch_num}, loss: {loss:.4f}")

    # Record the loss and val_loss for each epoch
    history['loss'].append(epoch_loss_avg.result().numpy())
    history['val_loss'].append(epoch_val_loss_avg.result().numpy())

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg.result().numpy()}, Val Loss: {epoch_val_loss_avg.result().numpy()}")

    # Early stopping check
    if epoch > 50 and epoch_val_loss_avg.result().numpy() >= min(history['val_loss'][-50:]):
        print("Early stopping triggered")
        break

plot_history(history)

# Evaluate the model on the test set
test_dataset = tf.data.Dataset.from_tensor_slices((input_test, output_test)).batch(batch_size)
test_loss_avg = tf.keras.metrics.Mean()

for x_batch_test, y_batch_test in test_dataset:
    y_pred_test = pinn_model(x_batch_test, training=False)
    test_loss = combined_loss(y_batch_test, y_pred_test)
    test_loss_avg.update_state(test_loss)

print(f"The final test loss is: {test_loss_avg.result().numpy()}")
print("The program has finished. Press Enter to exit.")
input()
