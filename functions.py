import os
import sys
import time
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- Build the model --------------------------- #


def build_pinn_model(input_shape=2, num_neurons=100, num_layers=4, output_shape=2):
    class PINNModel(nn.Module):
        def __init__(self):
            super(PINNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(input_shape, num_neurons))
            layers.append(nn.Tanh())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(num_neurons, num_neurons))
                # layers.append(nn.BatchNorm1d(num_neurons))
                layers.append(nn.Tanh())
                # layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(num_neurons, output_shape))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = PINNModel()
    return model


# -------------------------- Train loss function -------------------------- #
def train_loss(fiber_batch, pinn_model, parameters):

    A_mse_term = supervised_loss(fiber_batch, pinn_model)

    return A_mse_term  # Return the total loss


# ----------------------- NLSE Residual Function (PyTorch) ----------------------- #
def nlse_residual_pytorch(fiber_batch, pinn_model, parameters):
    # Unpack input data
    z = fiber_batch[:, 0:1].requires_grad_(True)
    t = fiber_batch[:, 1:2].requires_grad_(True)

    # Extract parameters and ensure they are tensors
    alpha = torch.tensor(parameters['alpha'], dtype=torch.float32)
    beta2 = torch.tensor(parameters['beta2'], dtype=torch.float32)
    gamma = torch.tensor(parameters['gamma'], dtype=torch.float32)

    # Forward pass through the PINN model
    inputs = torch.cat([z, t], dim=1)
    a_pred = pinn_model(inputs)
    a_pred_real = a_pred[:, 0:1]
    a_pred_imag = a_pred[:, 1:2]
    a_pred_complex = torch.complex(a_pred_real, a_pred_imag)

    # First derivatives with respect to t
    a_t_real = torch.autograd.grad(outputs=a_pred_real, inputs=t, grad_outputs=torch.ones_like(a_pred_real),
                                   create_graph=True, retain_graph=True)[0]
    a_t_imag = torch.autograd.grad(outputs=a_pred_imag, inputs=t, grad_outputs=torch.ones_like(a_pred_imag),
                                   create_graph=True, retain_graph=True)[0]

    # First derivatives with respect to z
    a_z_real = torch.autograd.grad(outputs=a_pred_real, inputs=z, grad_outputs=torch.ones_like(a_pred_real),
                                   create_graph=True, retain_graph=True)[0]
    a_z_imag = torch.autograd.grad(outputs=a_pred_imag, inputs=z, grad_outputs=torch.ones_like(a_pred_imag),
                                   create_graph=True, retain_graph=True)[0]

    # Second derivatives with respect to t
    a_tt_real = torch.autograd.grad(outputs=a_t_real, inputs=t, grad_outputs=torch.ones_like(a_t_real),
                                    create_graph=True, retain_graph=True)[0]
    a_tt_imag = torch.autograd.grad(outputs=a_t_imag, inputs=t, grad_outputs=torch.ones_like(a_t_imag),
                                    create_graph=True, retain_graph=True)[0]
    a_tt = torch.complex(a_tt_real, a_tt_imag)

    # Compute |A|^2
    a_pred_abs_squared = a_pred_real ** 2 + a_pred_imag ** 2  # Real-valued

    # Chromatic dispersion term
    beta2_c = beta2.type(torch.complex64)
    chrom_dis = (1j * beta2_c / 2) * a_tt

    # A_z complex term
    a_z = torch.complex(a_z_real, a_z_imag)

    # Attenuation
    alpha_c = alpha.type(torch.complex64)
    attenuation = (alpha_c / 2) * a_pred_complex

    # Nonlinear term
    gamma_c = gamma.type(torch.complex64)
    non_lin = gamma_c * a_pred_abs_squared * a_pred_complex

    # Residual calculation
    nlse_residual_value = a_z + chrom_dis + attenuation - non_lin


    # Residual real and imaginary parts
    nlse_residual_value_real = nlse_residual_value.real
    nlse_residual_value_imag = nlse_residual_value.imag

    # Compute the residual's magnitude
    nlse_term_real = torch.mean(nlse_residual_value_real ** 2)
    nlse_term_imag = torch.mean(nlse_residual_value_imag ** 2)

    return (nlse_term_real + nlse_term_imag).type(torch.float64)


# ---------------------- Initial Condition Loss Function ---------------------- #
def initial_condition_loss(A0_batch, pinn_model):
    z0 = A0_batch[:, 0:1]
    t0 = A0_batch[:, 1:2]
    A0_real = A0_batch[:, 2:3]
    A0_image = A0_batch[:, 3:4]

    inputs = torch.cat([z0, t0], dim=1)
    A0_pred = pinn_model(inputs)
    A0_pred_real = A0_pred[:, 0:1]
    A0_pred_image = A0_pred[:, 1:2]

    A0_real_mse = torch.mean((A0_pred_real - A0_real).abs() ** 2)
    A0_imag_mse = torch.mean((A0_pred_image - A0_image).abs() ** 2)

    A0_mse = A0_real_mse + A0_imag_mse

    return A0_mse


# ---------------------- Boundary Condition Loss Function ---------------------- #
def boundary_condition_loss(boundary_batch, pinn_model):
    zb = boundary_batch[:, 0:1]
    tb = boundary_batch[:, 1:2]
    Ab_real = boundary_batch[:, 2:3]
    Ab_image = boundary_batch[:, 3:4]

    inputs = torch.cat([zb, tb], dim=1)
    Ab_pred = pinn_model(inputs)
    Ab_pred_real = Ab_pred[:, 0:1]
    Ab_pred_imag = Ab_pred[:, 1:2]

    Ab_mse_real = torch.mean((Ab_pred_real - Ab_real).abs() ** 2)
    Ab_mse_imag = torch.mean((Ab_pred_imag - Ab_image).abs() ** 2)

    return Ab_mse_real + Ab_mse_imag


# ------------------------ Supervised Loss Function ------------------------ #
def supervised_loss(all_batch, pinn_model):
    z = all_batch[:, 0:1]
    t = all_batch[:, 1:2]
    A_real = all_batch[:, 2:3]
    A_image = all_batch[:, 3:4]

    inputs = torch.cat([z, t], dim=1)
    A_pred = pinn_model(inputs)
    A_pred_real = A_pred[:, 0:1]
    A_pred_imag = A_pred[:, 1:2]

    A_mse_real = torch.mean((A_pred_real - A_real).abs() ** 2)
    A_mse_imag = torch.mean((A_pred_imag - A_image).abs() ** 2)

    A_mse = A_mse_real + A_mse_imag
    return A_mse


# --------------------------- Test Loss Function --------------------------- #
def test_loss(batch, pinn_model):
    z_test = batch[:, 0:1]
    t_test = batch[:, 1:2]
    a_real = batch[:, 2:3]
    a_image = batch[:, 3:4]

    inputs = torch.cat([z_test, t_test], dim=1)
    a_pred = pinn_model(inputs)
    a_pred_real = a_pred[:, 0:1]
    a_pred_imag = a_pred[:, 1:2]

    test_loss_real = torch.mean((a_pred_real - a_real).abs() ** 2)
    test_loss_imag = torch.mean((a_pred_imag - a_image).abs() ** 2)

    testing_loss = test_loss_real + test_loss_imag
    return testing_loss


# --------------------------- Training Step Function --------------------------- #
def train_step(model, optimizer, input_train_batch, parameters):
    optimizer.zero_grad()
    loss = train_loss(input_train_batch, model, parameters)
    loss.backward()
    optimizer.step()
    return loss


# -------------------- Function to Plot Pulse Propagation -------------------- #
def plot_model_pulse_propagation(model, standardized_input, standardized_output, standardization_params, parameters):
    # Constants
    T0 = parameters['T0']  # Initial pulse width (ps)
    L = parameters['L']  # Fiber length (km)
    L_D = parameters['L_D']
    dz = parameters['dz']  # Step size in z (km)
    T = parameters['T']  # Time window (ps)
    Nt = parameters['Nt']  # Number of time points
    epochs = parameters['epochs']
    train_percentage = parameters['train_percentage']
    validation_percentage = parameters['validation_percentage']

    t = np.linspace(-T / 2, T / 2, Nt)
    z = np.linspace(0, L, int(L / dz))

    # Get predictions from the model
    model.eval()
    with torch.no_grad():
        standardized_input_tensor = torch.tensor(standardized_input, dtype=torch.float32)
        predictions = model(standardized_input_tensor).cpu().numpy()

    input_mean, input_std, output_mean, output_std = standardization_params

    # De-normalize the predicted output
    predictions_real = predictions[:, 0] * output_std[0] + output_mean[0]
    predictions_imag = predictions[:, 1] * output_std[1] + output_mean[1]
    predictions_complex = predictions_real + 1j * predictions_imag

    # De-normalize the SSFM output
    ssfm_real = standardized_output[:, 0] * output_std[0] + output_mean[0]
    ssfm_imag = standardized_output[:, 1] * output_std[1] + output_mean[1]
    ssfm_complex = ssfm_real + 1j * ssfm_imag

    A_t = predictions_complex.reshape(len(z), len(t))
    ssfm = ssfm_complex.reshape(len(z), len(t))

    os.makedirs('plots', exist_ok=True)
    params_str = (f"{epochs} epochs, {train_percentage*100}% train, "
                  f"{validation_percentage*100}% validation, {100-(train_percentage+validation_percentage)*100}% test")
    params_dir = os.path.join('plots', params_str)
    os.makedirs(params_dir, exist_ok=True)

    # Plot the results
    plt.figure(figsize=(20, 5))
    plt.imshow(np.abs(A_t).T, extent=[(z / L_D).min(), (z / L_D).max(), (t / T0).min(), (t / T0).max()], aspect='auto',
               origin='lower', cmap='jet')
    plt.colorbar(label='|A(z,t)|')
    plt.xlabel(f'Distance (z) | L_D = {round(L_D, 2)} Km')
    plt.ylabel(f'Time (t) | T0 = {T0} ps')
    plt.title(f'Pulse propagation using trained model |A(z,t)|')
    plt.savefig(os.path.join(params_dir, 'PINN_pulse_propagation.png'))
    plt.close()

    # 3D plot
    Z, T_grid = np.meshgrid(z / L_D, t / T0)
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, T_grid, np.abs(A_t.T), cmap='jet')
    ax.set_xlabel(f'Distance (z) | L_D = {round(L_D, 2)} Km')
    ax.set_ylabel(f'Time (t) | T0 = {T0} ps')
    ax.set_zlabel('|A(z,t)|')
    ax.set_title(
        f'3D view of pulse propagation using trained model |A(z,t)|')
    plt.savefig(os.path.join(params_dir, 'PINN_pulse_propagation_3D.png'))
    plt.close()

    # Initial and Final Pulse Comparison
    plt.figure(figsize=(20, 5))
    plt.plot(t, np.abs(A_t[0, :]), label='Initial Pulse Prediction')
    plt.plot(t, np.abs(ssfm[0, :]), label='Initial Pulse SSFM', linestyle='--')
    plt.plot(t, np.abs(A_t[-1, :]), label='Final Pulse Prediction')
    plt.plot(t, np.abs(ssfm[-1, :]), label='Final Pulse SSFM', linestyle='--')

    plt.xlabel(f'Time (ps) | T0 = {T0} ps')
    plt.ylabel('Amplitude (W)')
    plt.title('Initial and Final Pulse Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(params_dir, 'Initial and Final Pulse Comparison.png'))
    plt.close()

    # pulse cuts
    distances = [2, 2.5, 3, 4.0]  # The distances (in LD) to compare
    indices = [(np.abs(z - d * L_D)).argmin() for d in distances]  # Find the indices closest to the desired LD

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))  # Create a 2x2 subplot layout
    axes = axes.flatten()  # Flatten the 2x2 grid to iterate easily

    for i, dist in enumerate(distances):
        idx = indices[i]  # Index corresponding to the propagation distance

        axes[i].plot(t, np.abs(A_t[idx, :]), label=f'Pulse Prediction at {dist}LD')
        axes[i].plot(t, np.abs(ssfm[idx, :]), label=f'SSFM Pulse at {dist}LD', linestyle='--')

        axes[i].set_xlabel(f'Time (ps) | T0 = {T0} ps')
        axes[i].set_ylabel('Amplitude (W)')
        axes[i].set_title(f'Pulse Comparison at {dist}LD')
        axes[i].legend()
        axes[i].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(params_dir, 'Pulse_Comparison_3.25_4LD.png'))
    plt.close()


# ------------------- Function to Plot MSE Heatmap ------------------- #
def plot_mse_heatmap(pinn_model, standardized_input, standardized_output, z_grid, t_grid, standardization_params,
                     parameters):
    # Constants
    L_D = parameters['L_D']
    T0 = parameters['T0']
    epochs = parameters['epochs']
    train_percentage = parameters['train_percentage']
    validation_percentage = parameters['validation_percentage']

    # Get predictions from the model
    pinn_model.eval()
    with torch.no_grad():
        standardized_input_tensor = torch.tensor(standardized_input, dtype=torch.float32)
        predictions = pinn_model(standardized_input_tensor).cpu().numpy()

    input_mean, input_std, output_mean, output_std = standardization_params

    # De-normalize the predicted output
    predictions_real = predictions[:, 0] * output_std[0] + output_mean[0]
    predictions_imag = predictions[:, 1] * output_std[1] + output_mean[1]
    predictions_complex = predictions_real + 1j * predictions_imag

    # De-normalize the SSFM output
    ssfm_real = standardized_output[:, 0] * output_std[0] + output_mean[0]
    ssfm_imag = standardized_output[:, 1] * output_std[1] + output_mean[1]
    ssfm_complex = ssfm_real + 1j * ssfm_imag

    # Determine the correct reshape dimensions
    reshape_dim1 = z_grid.shape[0]  # Number of z points
    reshape_dim2 = t_grid.shape[1]  # Number of t points

    # Reshape the predictions and SSFM output to match the grid shape
    predictions_complex = predictions_complex.reshape(reshape_dim1, reshape_dim2)
    ssfm_complex = ssfm_complex.reshape(reshape_dim1, reshape_dim2)

    abs_error = np.abs(predictions_complex-ssfm_complex)

    os.makedirs('plots', exist_ok=True)
    params_str = (f"{epochs} epochs, {train_percentage * 100}% train, "
                  f"{validation_percentage * 100}% validation, {100 - (train_percentage + validation_percentage) * 100}% test")
    params_dir = os.path.join('plots', params_str)
    os.makedirs(params_dir, exist_ok=True)

    # Plot the Absolute Error heatmap
    plt.figure(figsize=(20, 5))
    plt.imshow(abs_error.T, extent=[z_grid.min()/L_D, z_grid.max()/L_D, t_grid.min()/T0, t_grid.max()/T0],
               aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=parameters['A0'])
    plt.colorbar(label='Absolute Error')
    plt.xlabel(f'Distance (L_D) | L_D = {L_D} km')
    plt.ylabel(f'Time (T0) | T0 = {T0} ps')
    plt.title('Absolute Error between A from PINN and A from SSFM')
    plt.savefig(os.path.join(params_dir, 'Absolute Error between A from PINN and A from SSFM.png'))
    plt.close()


# ----------------------- Function to Plot Training History ----------------------- #
def plot_history(history, parameters):

    # Constants
    epochs = parameters['epochs']
    train_percentage = parameters['train_percentage']
    validation_percentage = parameters['validation_percentage']

    os.makedirs('plots', exist_ok=True)
    params_str = (f"{epochs} epochs, {train_percentage * 100}% train, "
                  f"{validation_percentage * 100}% validation, {100 - (train_percentage + validation_percentage) * 100}% test")
    params_dir = os.path.join('plots', params_str)
    os.makedirs(params_dir, exist_ok=True)

    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.plot(history['test_loss'], label='Test')

    plt.yscale('log')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(params_dir, 'Training History.png'))
    plt.close()
