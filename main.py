
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model


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
    nlse_residual = A_pred_z + chrom_dis_complex + attenuation_complex - non_lin_complex
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


# Load the processed data
data = np.load('processed_training_data.npz')
input_train = data['input_train']
output_train = data['output_train']
input_val = data['input_val']
output_val = data['output_val']
input_test = data['input_test']
output_test = data['output_test']


# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# Load the best model parameters if available
checkpoint_path = "best_model.weights.h5"
try:
    pinn_model.load_weights(checkpoint_path)
    print("Loaded best model parameters from checkpoint.")
except:
    print("No checkpoint found, training from scratch.")


# Define the ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Training parameters
epochs = 5
batch_size = 1024

input_train = tf.cast(input_train, tf.float32)
output_train = tf.cast(output_train, tf.float32)
input_val = tf.cast(input_val, tf.float32)

# Prepare the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((input_train, output_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((input_val, output_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((input_test, output_test)).batch(batch_size)

# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
wait = 0

# Custom training loop
history = {'loss': [], 'val_loss': []}

print(f"Train dataset size: {len(list(train_dataset))} batches")
print(f"Validation dataset size: {len(list(val_dataset))} batches")
print(f"Test dataset size: {len(list(test_dataset))} batches")

for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()

    train_batch_num = 1
    val_batch_num = 1

    # Training loop
    for x_batch, y_batch in train_dataset:
        loss = train_step(pinn_model, optimizer, combined_loss, x_batch, y_batch)
        epoch_loss_avg.update_state(loss)
        train_batch_num += 1
        if (train_batch_num % 100) == 0:
            print(f"Training batch number {train_batch_num}/{len(list(train_dataset))}, Training loss: {loss:.4f}")

    # Validation loop
    for x_batch_val, y_batch_val in val_dataset:
        y_pred_val = pinn_model(x_batch_val, training=False)
        val_loss = combined_loss(y_batch_val, y_pred_val)
        epoch_val_loss_avg.update_state(val_loss)
        val_batch_num += 1
        if (val_batch_num % 50) == 0:
            print(f"Validation batch number {val_batch_num}/{len(list(val_dataset))}, Validation loss: {val_loss:.4f}")

    # Record the loss and val_loss for each epoch
    train_loss_value = epoch_loss_avg.result().numpy()
    val_loss_value = epoch_val_loss_avg.result().numpy()

    history['loss'].append(epoch_loss_avg.result().numpy())
    history['val_loss'].append(epoch_val_loss_avg.result().numpy())

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg.result().numpy()}, Val Loss: {epoch_val_loss_avg.result().numpy()}")

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

plot_history(history)

# Evaluate the model on the test set

test_loss_avg = tf.keras.metrics.Mean()

for x_batch_test, y_batch_test in test_dataset:
    y_pred_test = pinn_model(x_batch_test, training=False)
    test_loss = combined_loss(y_batch_test, y_pred_test)
    test_loss_avg.update_state(test_loss)

print(f"The final test loss is: {test_loss_avg.result().numpy()}")
print("The program has finished. Press Enter to exit.")
input()
