import torch
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt


cmap = 'viridis'

parser = argparse.ArgumentParser(description="Combine the results from the different job arrays")
parser.add_argument("--folder_name", type=str, default="", help="Name of the folder")
args = parser.parse_args()

# Define the path and folder name
folder_name = args.folder_name
path = os.path.join('/scratch/sr6364/perturbed_organics', 'data', folder_name)

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
spectral_radius = torch.full((num_delta, num_input, num_trials), fill_value=float('nan'), dtype=torch.float32)
norm_fixed_point_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float32)
norm_fixed_point_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float32)
actual_fixed_point_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float32)
actual_fixed_point_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float32)


# Load and combine partial results
for task_id in range(num_tasks):
    bool_stable_task = torch.load(os.path.join(path, f'bool_stable_task_{task_id}.pt'))
    spectral_radius_task = torch.load(os.path.join(path, f'spectral_radius_task_{task_id}.pt'))
    norm_fixed_point_y_task = torch.load(os.path.join(path, f'norm_fixed_point_y_task_{task_id}.pt'))
    norm_fixed_point_a_task = torch.load(os.path.join(path, f'norm_fixed_point_a_task_{task_id}.pt'))
    actual_fixed_point_y_task = torch.load(os.path.join(path, f'actual_fixed_point_y_task_{task_id}.pt'))
    actual_fixed_point_a_task = torch.load(os.path.join(path, f'actual_fixed_point_a_task_{task_id}.pt'))

    # Find indices where bool_stable_task != -1
    indices = (bool_stable_task != -1).nonzero(as_tuple=True)
    bool_stable[indices] = bool_stable_task[indices]
    spectral_radius[indices] = spectral_radius_task[indices]
    norm_fixed_point_y[indices] = norm_fixed_point_y_task[indices]
    norm_fixed_point_a[indices] = norm_fixed_point_a_task[indices]
    actual_fixed_point_y[indices] = actual_fixed_point_y_task[indices]
    actual_fixed_point_a[indices] = actual_fixed_point_a_task[indices]

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

# Calculate ratio norm differences
y_ratio = torch.norm(norm_fixed_point_y - actual_fixed_point_y, dim=3) / torch.norm(norm_fixed_point_y, dim=3)
a_ratio = torch.norm(norm_fixed_point_a - actual_fixed_point_a, dim=3) / torch.norm(norm_fixed_point_a, dim=3)


# Plot the heatmap
plt.figure(figsize=(12, 10))
plt.imshow(percent_stable, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap='viridis')
colorbar = plt.colorbar(label="Percent Stable Circuits", fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=20)

# Set font sizes
plt.xlabel('Input norm', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22)
plt.title("Phase Diagram: Percent Stable Circuits", fontsize=20)

# Increase tick label size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
colorbar.set_label("Percent Stable Circuits", fontsize=20)
# save the figure in svg
save_fig_path = os.path.join(path, 'percent_stable_circuits.svg')
plt.savefig(save_fig_path)

# spectral radius plotting
spectral_radius_mean = spectral_radius.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(spectral_radius_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap)
colorbar = plt.colorbar(label="Mean Spectral Radius", fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=14)

# Set the labels and title
plt.xlabel('Input norm', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22)
plt.title("Phase Diagram: Mean Spectral Radius", fontsize=20)

# Increase tick label size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
colorbar.set_label("Mean Spectral Radius", fontsize=20)

# Save the figure as SVG
save_fig_path = os.path.join(path, 'mean_spectral_radius.svg')
plt.savefig(save_fig_path)


# Plot the y_ratio heatmap
y_ratio_mean = y_ratio.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(y_ratio_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap)
colorbar = plt.colorbar(label="Ratio", fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=20)
plt.xlabel('Input norm', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22)
plt.title("Mean Ratio of Norm Difference for y", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(os.path.join(path, 'mean_y_ratio.svg'))

# Plot the a_ratio heatmap
a_ratio_mean = a_ratio.mean(dim=2)
plt.figure(figsize=(12, 10))
plt.imshow(a_ratio_mean, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap=cmap)
colorbar = plt.colorbar(label="Ratio", fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=20)
plt.xlabel('Input norm', fontsize=22)
plt.ylabel(r'$\Delta$', fontsize=22)
plt.title("Mean Ratio of Norm Difference for a", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(os.path.join(path, 'mean_a_ratio.svg'))
