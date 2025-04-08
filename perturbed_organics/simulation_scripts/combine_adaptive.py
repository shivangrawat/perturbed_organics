import torch
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


cmap = 'plasma'


parser = argparse.ArgumentParser(description="Combine the results from the different job arrays")
parser.add_argument("--folder_loc", type=str, default="", help="Location of the folder")
parser.add_argument("--folder_name", type=str, default="", help="Name of the folder")
args = parser.parse_args()

# Define the path and folder name
folder_loc = args.folder_loc
folder_name = args.folder_name
path = os.path.join(folder_loc, folder_name)

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
condition = torch.full((num_delta, num_input, num_trials), fill_value=-1, dtype=torch.int8)
spectral_radius = torch.full((num_delta, num_input, num_trials), fill_value=float('nan'), dtype=torch.float16)
norm_fixed_point_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
norm_fixed_point_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
actual_fixed_point_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
actual_fixed_point_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
first_order_perturb_y = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
first_order_perturb_a = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
eigvals_J = torch.full((num_delta, num_input, num_trials, 2 * N), fill_value=float("nan"), dtype=torch.complex64)

print(f"Combining partial results from {num_tasks} tasks")

# Load and combine partial results
for task_id in range(num_tasks):
    # Check if the file for this task exists before trying to load
    condition_file_path = os.path.join(path, f'condition_task_{task_id}.pt')
    if os.path.exists(condition_file_path):
        try:
            condition_task = torch.load(condition_file_path)
            spectral_radius_task = torch.load(os.path.join(path, f'spectral_radius_task_{task_id}.pt'))
            norm_fixed_point_y_task = torch.load(os.path.join(path, f'norm_fixed_point_y_task_{task_id}.pt'))
            norm_fixed_point_a_task = torch.load(os.path.join(path, f'norm_fixed_point_a_task_{task_id}.pt'))
            actual_fixed_point_y_task = torch.load(os.path.join(path, f'actual_fixed_point_y_task_{task_id}.pt'))
            actual_fixed_point_a_task = torch.load(os.path.join(path, f'actual_fixed_point_a_task_{task_id}.pt'))
            first_order_perturb_y_task = torch.load(os.path.join(path, f'first_order_perturb_y_task_{task_id}.pt'))
            first_order_perturb_a_task = torch.load(os.path.join(path, f'first_order_perturb_a_task_{task_id}.pt'))
            eigvals_J_task = torch.load(os.path.join(path, f'eigvals_J_task_{task_id}.pt'))

            # Find indices where condition_task != -1 (i.e., where the task actually computed results)
            indices = (condition_task != -1).nonzero(as_tuple=True)
            condition[indices] = condition_task[indices]
            spectral_radius[indices] = spectral_radius_task[indices]
            norm_fixed_point_y[indices] = norm_fixed_point_y_task[indices]
            norm_fixed_point_a[indices] = norm_fixed_point_a_task[indices]
            actual_fixed_point_y[indices] = actual_fixed_point_y_task[indices]
            actual_fixed_point_a[indices] = actual_fixed_point_a_task[indices]
            first_order_perturb_y[indices] = first_order_perturb_y_task[indices]
            first_order_perturb_a[indices] = first_order_perturb_a_task[indices]
            eigvals_J[indices] = eigvals_J_task[indices]
        except FileNotFoundError:
            # This handles cases where some but not all files for a task_id might be missing
            print(f"Warning: Some files for task_id {task_id} were missing. Skipping this task.")
            continue # Skip to the next task_id
    else:
        # If the condition file doesn't exist, assume the task failed and skip it.
        # The corresponding entries in the full result tensors will remain NaN/ -1.
        print(f"Task {task_id} results not found, skipping.")


# remove all the files ending with .pt inside the folder
for file in os.listdir(path):
    if file.endswith('.pt'):
        os.remove(os.path.join(path, file))


# Save full results
torch.save(condition, os.path.join(path, f'condition.pt'))
torch.save(spectral_radius, os.path.join(path, f'spectral_radius.pt'))
torch.save(norm_fixed_point_y, os.path.join(path, f'norm_fixed_point_y.pt'))
torch.save(norm_fixed_point_a, os.path.join(path, f'norm_fixed_point_a.pt'))
torch.save(actual_fixed_point_y, os.path.join(path, f'actual_fixed_point_y.pt'))
torch.save(actual_fixed_point_a, os.path.join(path, f'actual_fixed_point_a.pt'))
torch.save(first_order_perturb_y, os.path.join(path, f'first_order_perturb_y.pt'))
torch.save(first_order_perturb_a, os.path.join(path, f'first_order_perturb_a.pt'))
torch.save(eigvals_J, os.path.join(path, f'eigvals_J.pt'))

print("Results saved successfully")