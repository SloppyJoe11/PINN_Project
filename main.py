
from functions import *


# ------------------------ Load the processed data------------------------ #
print('Loading Data...')
data = np.load('processed_training_data.npz')
train_data = data['train_data']
val_data = data['val_data']
test_data = data['test_data']

standardized_combined_data = data['standardized_combined_data']
standardization_params = data['standardization_params']
standardized_input_data = data['standardized_input_data']
standardized_output_data = data['standardized_output_data']
z_grid = data['Z_grid']
t_grid = data['T_grid']
print('Data loaded!!!')

# ----------------------------- Parameters setting --------------------------------#

pinn_model = build_pinn_model()  # Create the network (should be a PyTorch model)

# Load the parameters dictionary from a file
with open('parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

# Training parameters
epochs = int(input('Number of epochs: '))
parameters['epochs'] = epochs
batch_size = 128

# Early stopping parameters
patience = int(0.2 * epochs)  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
wait = 0


# ----------------- -------- Data preparation --------------------------- #

# Prepare the validation and test datasets (without shuffling)
val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data, dtype=torch.float32))

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


buffer_size_train = len(train_data)


# ---------------------------- Optimizer and Checkpoint ---------------------- #
# Define optimizer
optimizer = torch.optim.Adam(pinn_model.parameters())
checkpoint_path = "best_model.pth"

print('Load latest model parameters?')
choice = input(' y/n: ')
if choice == 'y':
    # Load the best model parameters if available
    if os.path.exists(checkpoint_path):
        pinn_model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded best model parameters from checkpoint.")
    else:
        print("No checkpoint found, training from scratch.")
else:
    print('Starting new training process.')

history = {'train_loss': [], 'val_loss': [], 'test_loss': []}

# ---------------------------- Pre-Train model test ------------------------------- #

test_loss_avg_start = []
pinn_model.eval()

with torch.no_grad():
    for test_batch in test_loader:
        test_loss_term = test_loss(test_batch[0], pinn_model)  # Assuming test_loss accepts data and model
        test_loss_avg_start.append(test_loss_term.item())

test_loss_start = np.mean(test_loss_avg_start)
print(f'Test loss before training: {test_loss_start}')

# ----------------------------- Epoch Loop ----------------------------#

for epoch in range(epochs):
    start_time = time.time()

    epoch_loss_avg = []
    epoch_val_loss_avg = []
    epoch_test_loss_avg = []

    # ----------------------- Shuffling and batching -----------------#
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_batch_num = 1
    val_batch_num = 1

    total_batches = len(train_loader)

    pinn_model.train()
    # ----------------------------------- Training loop ---------------------------- #
    for train_batch in train_loader:

        loss = train_step(pinn_model, optimizer, train_batch[0], parameters)
        epoch_loss_avg.append(loss.item())

        # Print loading bar
        progress = train_batch_num / total_batches
        bar = f"[{'=' * int(progress * 40):40s}] {int(progress * 100)}%"
        sys.stdout.write(f"\rEpoch {epoch + 1}/{epochs} {bar}")
        sys.stdout.flush()

        train_batch_num += 1

    pinn_model.eval()
    # Validation loop
    for val_batch in val_loader:
        val_loss = train_loss(val_batch[0], pinn_model, parameters)
        epoch_val_loss_avg.append(val_loss.item())
        val_batch_num += 1

    # Testing loop
    for test_batch in test_loader:
        test_loss_term = test_loss(test_batch[0], pinn_model)
        epoch_test_loss_avg.append(test_loss_term.item())

    # Record the loss and val_loss for each epoch
    train_loss_value = np.mean(epoch_loss_avg)
    val_loss_value = np.mean(epoch_val_loss_avg)
    test_loss_value = np.mean(epoch_test_loss_avg)

    history['train_loss'].append(train_loss_value)
    history['val_loss'].append(val_loss_value)
    history['test_loss'].append(test_loss_value)

    print()
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss_value},"
          f" Val Loss: {val_loss_value},"
          f" Test Loss: {test_loss_value}")

    # Save the best model parameters
    if val_loss_value < best_val_loss:
        best_val_loss = val_loss_value
        wait = 0  # Reset wait counter
        torch.save(pinn_model.state_dict(), checkpoint_path)
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


np.savez('history.npz', history=history)
print('Saved history to history.npz')

history = np.load('history.npz', allow_pickle=True)
history = history['history'].item()

plot_history(history, parameters)

# Evaluate the model on the test set

test_loss_avg = []
pinn_model.eval()
with torch.no_grad():
    for test_batch in test_loader:
        test_loss_term = test_loss(test_batch[0], pinn_model)
        test_loss_avg.append(test_loss_term.item())

test_loss_final = np.mean(test_loss_avg)
print(f"The starting test loss is: {test_loss_start}, the final test loss is: {test_loss_final}")

pinn_model.load_state_dict(torch.load(checkpoint_path))  # Load best weights

# Function to calculate MSE between A from PINN and A from SSFM and plot it as a heatmap
plot_model_pulse_propagation(pinn_model, standardized_input_data, standardized_output_data,
                             standardization_params, parameters)
plot_mse_heatmap(pinn_model, standardized_input_data, standardized_output_data, z_grid, t_grid,
                 standardization_params, parameters)
