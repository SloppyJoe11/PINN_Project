import os
import time
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow C++ warnings

# Parameters for the NLSE
beta_2 = -20e-28  # s^2/m  # TODO: ssfm was set to -28 instead of -27, need to fix it in data_gen
gamma = 1.27e-3  # 1/(W*m)
alpha = 0  # Convert from dB/km if needed, else use direct 1/m


# Build the model
def build_pinn_model(input_shape=2, num_neurons=100, num_layers=4, output_shape=2):
    inputs = Input(shape=(input_shape,))
    x = Dense(num_neurons, activation='tanh')(inputs)
    for _ in range(num_layers - 1):
        x = Dense(num_neurons, activation='tanh')(x)
    outputs = Dense(output_shape, activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Plot training history
def plot_history(history):
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.plot(history['test_loss'], label='Test')
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper right')
    plt.show()


# Train loss function
def train_loss(fiber_batch, A0_batch, boundary_batch):

    nlse_loss_term = nlse_residual(fiber_batch)

    A0_mse_term = initial_condition_loss(A0_batch)

    Ab_mse_term = boundary_condition_loss(boundary_batch)

    return nlse_loss_term + A0_mse_term + Ab_mse_term


def nlse_residual(fiber_batch):
    # Split fiber batch into their respective components z and t
    z, t, a_real, a_image = tf.split(fiber_batch, [1, 1, 1, 1], axis=-1)

    # NLSE residual
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([z, t])

        a_pred = pinn_model(tf.concat([z, t], axis=-1))
        a_pred_real, a_pred_imag = tf.split(a_pred, 2, axis=-1)

        # First-order gradients
        a_t_real = tape2.gradient(a_pred_real, t)
        a_t_imag = tape2.gradient(a_pred_imag, t)

    a_pred_complex = tf.complex(a_pred_real, a_pred_imag)

    # Combine first-order gradients into complex gradients
    a_z_real = tape2.gradient(a_pred_real, z)
    a_z_imag = tape2.gradient(a_pred_imag, z)

    a_z_complex = tf.cast(tf.complex(a_z_real, a_z_imag), tf.complex64)

    # Second-order gradients
    a_tt_real = tape2.gradient(a_t_real, t)
    a_tt_imag = tape2.gradient(a_t_imag, t)

    a_tt_complex = tf.complex(a_tt_real, a_tt_imag)

    del tape2  # Free up resources used by the persistent tape

    # Compute NLSE residual terms
    a_pred_abs_squared = tf.cast(tf.square(tf.abs(a_pred_complex)), tf.complex64)
    attenuation_complex = (alpha / 2) * a_pred_complex
    chrom_dis_complex = (1j * beta_2 / 2) * tf.cast(a_tt_complex, tf.complex64)
    non_lin_complex = 1j * gamma * a_pred_abs_squared * a_pred_complex

    # NLSE residual calculation
    nlse_residual = a_z_complex + chrom_dis_complex + attenuation_complex - non_lin_complex
    nlse_term = tf.reduce_mean(tf.square(tf.abs(nlse_residual)))

    return nlse_term


def initial_condition_loss(A0_batch):

    # Initial condition MSE ( MSE(A_pred-A_ssfm) )
    z0, t0, A0_real, A0_image = tf.split(A0_batch, [1, 1, 1, 1], axis=-1)
    A0_ssfm = tf.cast(tf.complex(A0_real, A0_image), tf.complex64)
    A0_pred = pinn_model(tf.concat([z0, t0], axis=1))
    A0_pred_real, A0_pred_image = tf.split(A0_pred, 2, axis=-1)
    A0_pred_complex = tf.complex(A0_pred_real, A0_pred_image)
    A0_mse = tf.reduce_mean(tf.square(tf.abs(A0_pred_complex - A0_ssfm)))

    return A0_mse


def boundary_condition_loss(boundary_batch):

    # Boundary condition MSE ( MSE(A_pred_boundary - A_ssfm_boundary) )
    zb, tb, Ab_real, Ab_image = tf.split(boundary_batch, [1, 1, 1, 1], axis=-1)
    Ab_ssfm = tf.cast(tf.complex(Ab_real, Ab_image), tf.complex64)
    Ab_pred = pinn_model(tf.concat([zb, tb], axis=-1))
    Ab_pred_real, Ab_pred_imag = tf.split(Ab_pred, 2, axis=-1)
    Ab_pred_complex = tf.complex(Ab_pred_real, Ab_pred_imag)
    Ab_mse = tf.reduce_mean(tf.square(tf.abs(Ab_pred_complex - Ab_ssfm)))

    return Ab_mse


def test_loss(batch):
    z_test, t_test, a_real, a_image = tf.split(batch, [1, 1, 1, 1], axis=-1)
    a_sffm = tf.cast(tf.complex(a_real, a_image), tf.complex64)

    a_pred = pinn_model(tf.concat([z_test, t_test], axis=-1))
    a_pred_real, a_pred_imag = tf.split(a_pred, 2, axis=-1)
    a_pred_complex = tf.complex(a_pred_real, a_pred_imag)

    testing_loss = tf.reduce_mean(tf.square(tf.abs(a_pred_complex-a_sffm)))
    return testing_loss


# Training step function
def train_step(model, optimizer, loss_fn, input_train_batch, A0_batch, boundary_batch):

    with tf.GradientTape() as tape:
        loss = loss_fn(input_train_batch, A0_batch, boundary_batch)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Main code
pinn_model = build_pinn_model()


# Load the processed data
print('Loading Data...')
data = np.load('processed_training_data.npz')
input_train = data['input_train']
output_train = data['output_train']
input_val = data['input_val']
output_val = data['output_val']
input_test = data['input_test']
output_test = data['output_test']
A0_train = data['A0_train']
A0_val = data['A0_val']
A_boundary_train = data['A_boundary_train']
A_boundary_val = data['A_boundary_val']
standardization_params = data['standardization_params']
print('Data loaded!!!')


# Combine input and output data for training
train_data = np.concatenate((input_train, output_train), axis=1)
val_data = np.concatenate((input_val, output_val), axis=1)
test_data = np.concatenate((input_test, output_test), axis=1)

# Define optimizer
optimizer = tf.keras.optimizers.Adam()
checkpoint_path = "best_model.weights.h5"

print('Load latest model parameters?')
choice = input(' y/n: ')
if choice == 'y':
    # Load the best model parameters if available
    try:
        pinn_model.load_weights(checkpoint_path)
        print("Loaded best model parameters from checkpoint.")
    finally:
        print("No checkpoint found, training from scratch.")
else:
    print('Starting new training process.')


# Define the ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                                         save_best_only=True, save_weights_only=True, verbose=1)

# Training parameters
epochs = 10
batch_size = 128
buffer_size_train = len(input_train)
buffer_size_A0 = len(A0_train)
buffer_size_boundary = len(A_boundary_train)


# Prepare the validation and test datasets (without shuffling)
val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)
A0_dataset_val = tf.data.Dataset.from_tensor_slices(A0_val).batch(batch_size, drop_remainder=True).repeat()
boundary_dataset_val = tf.data.Dataset.from_tensor_slices(A_boundary_val).batch(batch_size, drop_remainder=True).repeat()


# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
wait = 0

# Custom training loop
history = {'loss': [], 'val_loss': [], 'test_loss': []}

test_loss_avg_start = tf.keras.metrics.Mean()

for test_batch in test_dataset:
    test_loss_term = test_loss(test_batch)
    test_loss_avg_start.update_state(test_loss_term)
print(f'Test loss before training: {test_loss_avg_start.result().numpy()}')

for epoch in range(epochs):
    start_time = time.time()
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_test_loss_avg = tf.keras.metrics.Mean()

    # Prepare the dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size_train).batch(batch_size,
                                                                                                    drop_remainder=True)
    A0_dataset = tf.data.Dataset.from_tensor_slices(A0_train).shuffle(buffer_size_A0).batch(batch_size,
                                                                                       drop_remainder=True).repeat()
    boundary_dataset = tf.data.Dataset.from_tensor_slices(A_boundary_train).shuffle(buffer_size_boundary).batch(batch_size,
                                                                                                          drop_remainder=True).repeat()

    # Combine the datasets into one
    combined_train_dataset = tf.data.Dataset.zip((train_dataset, A0_dataset, boundary_dataset))
    combined_validation_dataset = tf.data.Dataset.zip((val_dataset, A0_dataset_val, boundary_dataset_val))

    train_batch_num = 1
    val_batch_num = 1

    # Training loop
    for train_batch, A0_batch, A_boundary_batch in combined_train_dataset:

        loss = train_step(pinn_model, optimizer, train_loss, train_batch, A0_batch, A_boundary_batch)
        epoch_loss_avg.update_state(loss)
        train_batch_num += 1
        if (train_batch_num % 100) == 0:
            print(f"Training batch number {train_batch_num}/{len(list(train_dataset))}, Training loss: {loss:.16f}")

    # Validation loop
    for val_batch, A0_batch, A_boundary_batch in combined_validation_dataset:

        val_loss = train_loss(val_batch, A0_batch, A_boundary_batch)
        epoch_val_loss_avg.update_state(val_loss)
        val_batch_num += 1
        if (val_batch_num % 100) == 0:
            print(
                f"Validation batch number {val_batch_num}/{len(list(val_dataset))}, Validation loss: {val_loss:.16f}")

    # Testing loop
    for test_batch in test_dataset:
        test_loss_term = test_loss(test_batch)
        epoch_test_loss_avg.update_state(test_loss_term)

    # Record the loss and val_loss for each epoch
    train_loss_value = epoch_loss_avg.result().numpy()
    val_loss_value = epoch_val_loss_avg.result().numpy()
    test_loss_value = epoch_test_loss_avg.result().numpy

    history['loss'].append(epoch_loss_avg.result().numpy())
    history['val_loss'].append(epoch_val_loss_avg.result().numpy())
    history['test_loss'].append(epoch_test_loss_avg.result().numpy())

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg.result().numpy()},"
          f" Val Loss: {epoch_val_loss_avg.result().numpy()},"
          f" Test Loss: {epoch_test_loss_avg.result().numpy()}")

    # Save the best model parameters
    if val_loss_value < best_val_loss:
        best_val_loss = val_loss_value
        wait = 0  # Reset wait counter
        pinn_model.save_weights(checkpoint_path)
        print("Saved best model parameters.")
    else:
        wait += 1
        print(f"Early stopping wait: {wait}/{patience}")

        if wait >= patience:
            print("Early stopping triggered")
            break

    # Early stopping check
    if epoch > 50 and epoch_val_loss_avg.result().numpy() >= min(history['val_loss'][-50:]):
        print("Early stopping triggered")
        break

    end_time = time.time()
    epoch_duration = end_time - start_time  # Calculate duration
    print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")


plot_history(history)

# Evaluate the model on the test set

test_loss_avg = tf.keras.metrics.Mean()

for test_batch in test_dataset:
    test_loss_term = test_loss(test_batch)
    test_loss_avg.update_state(test_loss_term)

print(f"The final test loss is: {test_loss_avg.result().numpy()},"
      f" the starting test loss is: {test_loss_avg_start.result().numpy()}")
print("The program has finished. Press Enter to exit.")
input()
