from functions import *
from sklearn.model_selection import train_test_split

# Build the model
pinn_model = build_pinn_model()

# Model summary
pinn_model.summary()


# Parameters for the NLSE
beta_2 = -21.27  # GVD parameter in ps^2/km
gamma = 1.3  # Nonlinear parameter in 1/(W km)
alpha = 0.2  # Attenuation parameter in dB/km


# parameters for data generation   TODO: fix data generation function to fit the normal units
fiber_length = 1000  # meters
num_steps = 1024
dt = 1e-4  # seconds
dz = 0.1  # meters
beta_2 = -21.27e-27  # s^2/m
gamma = 1.3e-3  # 1/(W*m)
alpha = 0.046 / 1000  # Convert from dB/km if needed, else use direct 1/m

# Generate the training data
Z, T, A = generate_training_data(gaussian_pulse, fiber_length,
                                 num_steps, dt, dz, beta_2, gamma, alpha)

# Call the plotting function with the data
# plot_results(Z, T, A)


# Assume Z and T are 1D NumPy arrays from the generate_training_data function
# Z is of shape (num_z_steps,) and T is of shape (num_t_steps, )

# Create a 2D grid of Z and T values
Z_grid, T_grid = np.meshgrid(Z, T, indexing='ij')

# Flatten the grids to create a 2D array of shape (num_z_steps * num_t_steps, 2)
input_data = np.vstack((Z_grid.flatten(), T_grid.flatten())).T


# Flatten A to have the same shape as input_data
output_data = A.flatten()

# Ensure the output is a complex number
output_data = np.stack((output_data.real, output_data.imag), axis=-1)

# Normalize the input and output data
standardized_input_data, standardized_output_data, standardization_params = standardize_data(input_data, output_data)

# Create the loss function
loss_function = create_physics_informed_loss(pinn_model, beta_2, gamma, alpha)

# Compile the model with the custom loss function
pinn_model.compile(optimizer='adam', loss=loss_function)


# Split the dataset into training and (validation + test)
input_train, input_val_test, output_train, output_val_test = train_test_split(
    standardized_input_data,
    standardized_output_data,
    test_size=0.3,  # 30% for validation and testing
    random_state=42  # Seed for reproducibility
)

# Split the (validation + test) into validation and test sets
input_val, input_test, output_val, output_test = train_test_split(
    input_val_test,
    output_val_test,
    test_size=0.5,  # Split the remaining 30% into two halves: 15% for validation and 15% for testing
    random_state=42
)

# Define a callback for early stopping to prevent overfitting
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train the model
history = pinn_model.fit(
    input_train,
    output_train,
    validation_data=(input_val, output_val),
    epochs=1000,  # You can adjust this
    batch_size=32,  # And this, according to your dataset and resource capability
    callbacks=[early_stopping_callback]
)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# Evaluate the model on the test set
test_loss = pinn_model.evaluate(input_test, output_test)

print(f"The final test loss is: {test_loss}")

print("The program has finished. Press Enter to exit.")
input()


