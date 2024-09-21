from functions import *

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow C++ warnings

# ------------------------ Load the processed data------------------------ #
print('Loading Data...')
data = np.load('processed_training_data.npz')
train_data = data['train_data']
val_data = data['val_data']
test_data = data['test_data']
A0_train = data['A0_train']
A0_val = data['A0_val']
A_boundary_train = data['A_boundary_train']
A_boundary_val = data['A_boundary_val']

standardized_combined_data = data['standardized_combined_data']
standardization_params = data['standardization_params']
standardized_input_data = data['standardized_input_data']
standardized_output_data = data['standardized_output_data']
z_grid = data['Z_grid']
t_grid = data['T_grid']
print('Data loaded!!!')


# ----------------------------- Parameters setting --------------------------------#

pinn_model = build_pinn_model()  # Create the network

# Load the parameters dictionary from a file
with open('parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

# Training parameters
epochs = 5
parameters['epochs'] = epochs
batch_size = 128

# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
wait = 0

parameters["beta2"] = -20
parameters["gamma"] = 0
print(f'parameters: alpha = {parameters["alpha"]} beta = {parameters["beta2"]} gamma = {parameters["gamma"]}')
# ------------------------- Data preparation --------------------------- #

# Prepare the validation and test datasets (without shuffling)
val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)
A0_dataset_val = tf.data.Dataset.from_tensor_slices(A0_val).batch(batch_size, drop_remainder=True).repeat()
boundary_dataset_val = tf.data.Dataset.from_tensor_slices(A_boundary_val).batch(batch_size, drop_remainder=True).repeat()

# Joining validation dataset
combined_validation_dataset = tf.data.Dataset.zip((val_dataset, A0_dataset_val, boundary_dataset_val))

# Decreasing Dataset size to 50K samples
sampled_train_data_indices = np.random.choice(len(train_data), 50000, replace=False)
sampled_train_data = train_data[sampled_train_data_indices]

buffer_size_train = len(sampled_train_data)
buffer_size_A0 = len(A0_train)
buffer_size_boundary = len(A_boundary_train)

# ---------------------------- Optimizer and Checkpoint ---------------------- #
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
    except:
        print("No checkpoint found, training from scratch.")
else:
    print('Starting new training process.')

# Define the ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                                         save_best_only=True, save_weights_only=True, verbose=1)

history = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'nlse_loss': [], 'A0_loss': [], 'Ab_loss': []}

# ---------------------------- Pre-Train model test ------------------------------- #

test_loss_avg_start = tf.keras.metrics.Mean()
#
# for test_batch in test_dataset:
#     test_loss_term = test_loss(test_batch, pinn_model)
#     test_loss_avg_start.update_state(test_loss_term)
# print(f'Test loss before training: {test_loss_avg_start.result().numpy()}')
#
#
# plot_model_pulse_propagation(pinn_model, standardized_input_data, standardization_params, parameters)


# ----------------------------- Epoch Loop ----------------------------#

for epoch in range(epochs):
    start_time = time.time()

    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_test_loss_avg = tf.keras.metrics.Mean()

    epoch_nlse_loss_avg = tf.keras.metrics.Mean()
    epoch_A0_loss_avg = tf.keras.metrics.Mean()
    epoch_Ab_loss_avg = tf.keras.metrics.Mean()

    # ----------------------- Shuffling and batching -----------------#
    train_dataset = tf.data.Dataset.from_tensor_slices(sampled_train_data).shuffle(buffer_size_train).batch(batch_size,
                                                                                                    drop_remainder=True)
    A0_dataset = tf.data.Dataset.from_tensor_slices(A0_train).shuffle(buffer_size_A0).batch(batch_size,
                                                                                            drop_remainder=True).repeat()
    boundary_dataset = tf.data.Dataset.from_tensor_slices(A_boundary_train).shuffle(buffer_size_boundary).batch(
        batch_size, drop_remainder=True).repeat()

    # TODO: delete after test!!!!
    all_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size_train).batch(batch_size,
                                                                                                    drop_remainder=True)

    # Joining training dataset  TODO: change all_dataset back to train_dataset after test!!!
    combined_train_dataset = tf.data.Dataset.zip((train_dataset, A0_dataset, boundary_dataset))

    train_batch_num = 1
    val_batch_num = 1

    total_batches = len(list(train_dataset))

    # ----------------------------------- Training loop ---------------------------- #
    for train_batch, A0_batch, A_boundary_batch in combined_train_dataset:

        loss = train_step(pinn_model, optimizer, train_batch, A0_batch, A_boundary_batch, parameters,
                          epoch_nlse_loss_avg, epoch_A0_loss_avg, epoch_Ab_loss_avg)
        epoch_loss_avg.update_state(loss)

        # Print loading bar
        progress = train_batch_num / total_batches
        bar = f"[{'=' * int(progress * 40):40s}] {int(progress * 100)}%"
        sys.stdout.write(f"\rEpoch {epoch + 1}/{epochs} {bar}")
        sys.stdout.flush()

        train_batch_num += 1

    # Validation loop
    for val_batch, A0_batch, A_boundary_batch in combined_validation_dataset:
        val_loss = train_loss(val_batch, A0_batch, A_boundary_batch, pinn_model, parameters,
                              epoch_nlse_loss_avg, epoch_A0_loss_avg, epoch_Ab_loss_avg)
        epoch_val_loss_avg.update_state(val_loss)
        val_batch_num += 1

    # Testing loop
    for test_batch in test_dataset:
        test_loss_term = test_loss(test_batch, pinn_model)
        epoch_test_loss_avg.update_state(test_loss_term)

    # Record the loss and val_loss for each epoch
    train_loss_value = epoch_loss_avg.result().numpy()
    val_loss_value = epoch_val_loss_avg.result().numpy()
    test_loss_value = epoch_test_loss_avg.result().numpy()

    A0_loss_value = epoch_A0_loss_avg.result().numpy()
    Ab_loss_value = epoch_Ab_loss_avg.result().numpy()
    nlse_loss_value = epoch_nlse_loss_avg.result().numpy()

    history['train_loss'].append(train_loss_value)
    history['val_loss'].append(val_loss_value)
    history['test_loss'].append(test_loss_value)
    history['nlse_loss'].append(nlse_loss_value)
    history['A0_loss'].append(A0_loss_value)
    history['Ab_loss'].append(Ab_loss_value)

    print()
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss_value},"
          f" Val Loss: {val_loss_value},"
          f" Test Loss: {test_loss_value}")

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
            parameters["epochs"] = epoch
            break

    end_time = time.time()
    epoch_duration = end_time - start_time  # Calculate duration
    print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

    # plot_model_pulse_propagation(pinn_model, standardized_input_data, standardization_params, parameters)

np.savez('history.npz', history=history)
print('Saved history to history.npz')


history = np.load('history.npz', allow_pickle=True)
history = history['history'].item()

plot_history(history, parameters)

# Evaluate the model on the test set

test_loss_avg = tf.keras.metrics.Mean()

for test_batch in test_dataset:
    test_loss_term = test_loss(test_batch, pinn_model,)
    test_loss_avg.update_state(test_loss_term)

print(f"The final test loss is: {test_loss_avg.result().numpy()},"
      f" the starting test loss is: {test_loss_avg_start.result().numpy()}")

# Function to calculate MSE between A from PINN and A from SSFM and plot it as a heatmap
plot_model_pulse_propagation(pinn_model, standardized_input_data, standardization_params, parameters)
plot_mse_heatmap(pinn_model, standardized_input_data, standardized_output_data, z_grid, t_grid, standardization_params, parameters)
