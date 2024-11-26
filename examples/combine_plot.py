import torch
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


cmap = 'plasma'


parser = argparse.ArgumentParser(description="Combine the results from the different job arrays")
parser.add_argument("--folder_name", type=str, default="", help="Name of the folder")
args = parser.parse_args()

# Define the path and folder name
folder_name = args.folder_name
path = os.path.join('/vast/sr6364/perturbed_organics', 'data', folder_name)

# Load parameters from the JSON file
param_file_path = os.path.join(path, 'parameters.json')
with open(param_file_path, 'r') as f:
    params = json.load(f)

# Extract parameters
N = params['N']
num_tasks = params['num_tasks']
num_delta = params['num_delta']
num_input = params['num_input']
num_trials = params['num_trials']
delta_range = np.array(params['delta_range'])
input_range = np.array(params['input_range'])

# Initialize full results arrays
bool_stable = torch.full((num_delta, num_input, num_trials), fill_value=-1, dtype=torch.int8)
spectral_radius = torch.full((num_delta, num_input, num_trials), fill_value=float('nan'), dtype=torch.float16)
norm_fixed_point_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
norm_fixed_point_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
actual_fixed_point_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
actual_fixed_point_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
first_order_perturb_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
first_order_perturb_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
eigvals_J = torch.full((num_delta, num_input, num_trials, 2 * N), fill_value=float("nan"), dtype=torch.complex64)


# Load and combine partial results
for task_id in range(num_tasks):
    bool_stable_task = torch.load(os.path.join(path, f'bool_stable_task_{task_id}.pt'))
    spectral_radius_task = torch.load(os.path.join(path, f'spectral_radius_task_{task_id}.pt'))
    norm_fixed_point_y_task = torch.load(os.path.join(path, f'norm_fixed_point_y_task_{task_id}.pt'))
    norm_fixed_point_a_task = torch.load(os.path.join(path, f'norm_fixed_point_a_task_{task_id}.pt'))
    actual_fixed_point_y_task = torch.load(os.path.join(path, f'actual_fixed_point_y_task_{task_id}.pt'))
    actual_fixed_point_a_task = torch.load(os.path.join(path, f'actual_fixed_point_a_task_{task_id}.pt'))
    first_order_perturb_y_task = torch.load(os.path.join(path, f'first_order_perturb_y_task_{task_id}.pt'))
    first_order_perturb_a_task = torch.load(os.path.join(path, f'first_order_perturb_a_task_{task_id}.pt'))
    eigvals_J_task = torch.load(os.path.join(path, f'eigvals_J_task_{task_id}.pt'))

    # Find indices where bool_stable_task != -1
    indices = (bool_stable_task != -1).nonzero(as_tuple=True)
    bool_stable[indices] = bool_stable_task[indices]
    spectral_radius[indices] = spectral_radius_task[indices]
    norm_fixed_point_y[indices] = norm_fixed_point_y_task[indices]
    norm_fixed_point_a[indices] = norm_fixed_point_a_task[indices]
    actual_fixed_point_y[indices] = actual_fixed_point_y_task[indices]
    actual_fixed_point_a[indices] = actual_fixed_point_a_task[indices]
    first_order_perturb_y[indices] = first_order_perturb_y_task[indices]
    first_order_perturb_a[indices] = first_order_perturb_a_task[indices]
    eigvals_J[indices] = eigvals_J_task[indices]

# check if all the files were merged properly
assert torch.all(bool_stable != -1)
assert torch.all(~torch.isnan(spectral_radius))

# remove all the files ending with .pt inside the folder
for file in os.listdir(path):
    if file.endswith('.pt'):
        os.remove(os.path.join(path, file))

# Proceed with plotting or further analysis
bool_stable = bool_stable.bool()
percent_stable = bool_stable.float().mean(dim=2) * 100

# Save full results
torch.save(bool_stable, os.path.join(path, f'bool_stable.pt'))
torch.save(spectral_radius, os.path.join(path, f'spectral_radius.pt'))
torch.save(percent_stable, os.path.join(path, f'percent_stable.pt'))
torch.save(norm_fixed_point_y, os.path.join(path, f'norm_fixed_point_y.pt'))
torch.save(norm_fixed_point_a, os.path.join(path, f'norm_fixed_point_a.pt'))
torch.save(actual_fixed_point_y, os.path.join(path, f'actual_fixed_point_y.pt'))
torch.save(actual_fixed_point_a, os.path.join(path, f'actual_fixed_point_a.pt'))
torch.save(first_order_perturb_y, os.path.join(path, f'first_order_perturb_y.pt'))
torch.save(first_order_perturb_a, os.path.join(path, f'first_order_perturb_a.pt'))
torch.save(eigvals_J, os.path.join(path, f'eigvals_J.pt'))


### Plot percent stable circuits ###
plt.figure(figsize=(12, 10))
norm = mcolors.Normalize(vmin=0, vmax=100, clip=False)
plt.imshow(percent_stable, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"% stable", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("Phase Diagram: Percent Stable Circuits", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'percent_stable_circuits.png'), bbox_inches='tight')


### spectral radius plotting ###
spectral_radius_mean = spectral_radius.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(spectral_radius_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"$\rho(\mathbf{W}_r)$", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("Phase Diagram: Mean Spectral Radius", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'mean_spectral_radius.png'), bbox_inches='tight')


### Plot the y_norm heatmap ###
norm = mcolors.Normalize(vmin=0, vmax=1, clip=False)
y_mean = torch.norm(actual_fixed_point_y, dim=3).mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(y_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"$||\mathbf{y}||$", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("Norm of y", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'y_norm.png'), bbox_inches='tight')


### Plot the a_norm heatmap ###
norm = mcolors.Normalize(vmin=0, vmax=1, clip=False)
a_mean = torch.norm(actual_fixed_point_a, dim=3).mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(a_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"$||\mathbf{a}||$", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("Norm of a", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'a_norm.png'), bbox_inches='tight')


### Plot the y_ratio heatmap ###
norm = mcolors.Normalize(vmin=0, vmax=1, clip=False)
y_ratio = torch.norm(norm_fixed_point_y - actual_fixed_point_y, dim=3) / torch.norm(norm_fixed_point_y, dim=3)
y_ratio_mean = y_ratio.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(y_ratio_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"$\frac{||\mathbf{y}-\mathbf{y}_0||}{||\mathbf{y}_0||}$", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("Actual Mean Ratio of Norm Difference for y", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'actual_mean_y_ratio.png'), bbox_inches='tight')


### Plot the a_ratio heatmap ###
norm = mcolors.Normalize(vmin=0, vmax=1, clip=False)
a_ratio = torch.norm(norm_fixed_point_a - actual_fixed_point_a, dim=3) / torch.norm(norm_fixed_point_a, dim=3)
a_ratio_mean = a_ratio.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(a_ratio_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"$\frac{||\mathbf{a}-\mathbf{a}_0||}{||\mathbf{a}_0||}$", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("Actual Mean Ratio of Norm Difference for a", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'actual_mean_a_ratio.png'), bbox_inches='tight')


### Plot the perturbed y_ratio ###
norm = mcolors.Normalize(vmin=0, vmax=1, clip=False)
y_ratio = torch.norm(first_order_perturb_y, dim=3) / torch.norm(norm_fixed_point_y, dim=3)
y_ratio_mean = y_ratio.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(y_ratio_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"$\frac{||\mathbf{y}_1||}{||\mathbf{y}_0||}$", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("First Order Mean Ratio of Norm Difference for y", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'first_order_mean_y_ratio.png'), bbox_inches='tight')


### Plot the perturbed a_ratio ###
norm = mcolors.Normalize(vmin=0, vmax=1, clip=False)
a_ratio = torch.norm(first_order_perturb_a, dim=3) / torch.norm(norm_fixed_point_a, dim=3)
a_ratio_mean = a_ratio.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(a_ratio_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"$\frac{||\mathbf{a}_1||}{||\mathbf{a}_0||}$", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("First Order Mean Ratio of Norm Difference for a", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'first_order_mean_a_ratio.png'), bbox_inches='tight')


### Plot the maximum real part of the eigenvalues of J ###
norm = mcolors.Normalize(vmin=None, vmax=0, clip=False)
eigvals_J_real = torch.max(torch.real(eigvals_J), dim=3).values
eigvals_J_real_mean = eigvals_J_real.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(eigvals_J_real_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)
colorbar = plt.colorbar(fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)
colorbar.ax.set_ylabel(r"Re($\lambda_J$)", fontsize=22, rotation=0, labelpad=30)
plt.xlabel('Input drive', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22, rotation=0, labelpad=15)
plt.title("Mean Maximum Real Part of Eigenvalues of J", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(path, 'mean_max_real_part_eigvals_J.png'), bbox_inches='tight')

