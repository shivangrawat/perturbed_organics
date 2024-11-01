import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import math
from perturbed_organics.spectrum_general import matrix_solution
from perturbed_organics.spectrum_general import sim_solution
import perturbed_organics.model.ORGaNICs_models as organics
from perturbed_organics.utils.util_funs import dynm_fun
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import argparse
import os
from matplotlib import colors
from scipy.optimize import fsolve, curve_fit
from scipy import integrate
import warnings


cmap = 'viridis'

# Import the arguments
parser = argparse.ArgumentParser(description="Sparse Matrix Stability Scan")

# arguments for the job array code
parser.add_argument('--TASK_ID', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)), help="Task ID")
parser.add_argument('--NUM_TASKS', type=int, default=1, help="Number of tasks")

# parameters of the model 
parser.add_argument('--MODEL_NAME', type=str, default="localized", help='Model name')
parser.add_argument('--N', type=int, default=100, help="Number of neurons")
parser.add_argument('--c', type=int, default=100, help="Sparsity")
parser.add_argument('--mu', type=float, default=0.0, help="Mean of the distribution")
parser.add_argument('--sigma', type=float, default=0.1, help="Sigma value")
parser.add_argument('--b0', type=float, default=0.5, help="Input gain for a")
parser.add_argument('--b1', type=float, default=0.5, help="Input gain for y")
parser.add_argument('--tauA', type=float, default=0.002, help="Time constant fot a")
parser.add_argument('--tauY', type=float, default=0.002, help="Time constant for y")

# parameters of simulation
parser.add_argument('--num_trials', type=int, default=10, help="Number of trials")
parser.add_argument('--num_delta', type=int, default=10, help="Number of delta steps")
parser.add_argument('--num_input', type=int, default=10, help="Number of input steps")
parser.add_argument('--max_delta', type=float, default=5.0, help="Maximum value of the parameter delta")
parser.add_argument('--max_input', type=float, default=5.0, help="Maximum value of the parameter input")

# parse the arguments
args = parser.parse_args()

# argument sof the job array
task_id = args.TASK_ID
num_tasks = args.NUM_TASKS

# arguments of the model parameters
model_name = args.MODEL_NAME
N = args.N
c = args.c
mu = args.mu

# arguments of the simulation
num_delta = args.num_delta
num_input = args.num_input
num_trials = args.num_trials
max_input = args.max_input
max_delta = args.max_delta


device = torch.device("cpu")

# Define the parameters of the ORGaNICs
params = {
    'N_y': N,
    'N_a': N,
    'eta': 0.02,
    'noise_type': 'additive'
}
b0 = args.b0 * torch.ones(N)
b1 = args.b1 * torch.ones(N)
sigma = torch.tensor([args.sigma])
tauA = args.tauA + 0 * torch.abs(torch.randn(N) * 0.001)
tauY = args.tauY + 0 * torch.abs(torch.randn(N) * 0.001)
# Wyy = torch.eye(N)
Way = torch.ones(N, N)


def sample_sparse_matrix(N, c, delta, mu):
    mask = torch.bernoulli(torch.full((N, N), c / N)).triu()
    values = torch.normal(mu / c, delta / math.sqrt(c), (N, N))
    upper_triangular = values * mask
    symmetric_matrix = upper_triangular + upper_triangular.T - torch.diag(torch.diag(upper_triangular))
    return symmetric_matrix


# Define the scan parameters
delta_range = np.linspace(0, max_delta, num_delta)
input_range = np.linspace(0.01, max_input, num_input)

# define the quantities that we'll calculate
spectral_radius = torch.zeros((num_delta, num_input, num_trials), dtype=torch.float32)
bool_stable = torch.zeros((num_delta, num_input, num_trials), dtype=torch.bool)

def run_trial(i, j, k, delta, input):
    # Initialize variables for the trial

    # make input one hot
    z = torch.zeros(N) + 1e-3
    z[0] = input

    Wyy = torch.eye(N) + sample_sparse_matrix(N, c, delta, mu)

    # find the spectral radius of Wyy 
    eigvals = torch.linalg.eigvals(Wyy)
    spectral_radius = torch.max(torch.abs(eigvals))

    # Start the simulation from the normalization fixed point
    a_s = sigma ** 2 * b0 ** 2 + Way @ (b1 * z) ** 2
    y_s = (b1 * z) / torch.sqrt(a_s)
    initial_sim = torch.cat((y_s, a_s), dim=0)

    # Instantiate the model
    model = organics.ORGaNICs2Dgeneral(
        params=params, b0=b0, b1=b1, sigma=sigma,
        tauA=tauA, tauY=tauY, Wyy=Wyy, Way=Way, z=z,
        initial_sim=initial_sim, run_jacobian=True
    )
    if model.J is None:
        return (False, spectral_radius)
    else:
        return (True, spectral_radius)

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_trial)(i, j, k, delta, input)
    for i, delta in enumerate(delta_range)
    for j, input in enumerate(input_range)
    for k in range(num_trials)
)

for idx, (i, j, k) in enumerate(
    (i, j, k) for i in range(num_delta) for j in range(num_input) for k in range(num_trials)
):
    bool_stable[i, j, k] = results[idx][0]
    spectral_radius[i, j, k] = results[idx][1]

# create a folder in ..data/ to save the results
folder_name = model_name + '_N_{}_c_{}_mu_{}_num_delta_{}_num_input_{}_num_trials_{}_b0_{}_b1_{}'.format(N, c, mu, num_delta, num_input, num_trials, args.b0, args.b1)
path = os.path.join('..', 'data', folder_name)

if not os.path.exists(path):
    os.makedirs(path)

torch.save(bool_stable, os.path.join(path, 'bool_stable.pt'))
torch.save(spectral_radius, os.path.join(path, 'spectral_radius.pt'))


# plot the circuit
percent_stable = bool_stable.float().mean(dim=2) * 100

# Plot the heatmap
plt.figure(figsize=(12, 10))
plt.imshow(percent_stable, extent=[input_range.min(), input_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', aspect='auto', cmap='viridis')
colorbar = plt.colorbar(label="Percent Stable Circuits", fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=14)

# plotting curve from flaviano
# plt.scatter(data_sim[:, 1], data_sim[:, 0], c='black', edgecolor='black', s=50, label="Simulation")
# plt.scatter(data_y[:, 0], data_y[:, 1], c='cyan', edgecolor='black', s=50, label="y-theory")
# plt.scatter(data_a[:, 0], data_a[:, 1], c='red', edgecolor='black', s=50, label="a-theory")

# add legend
plt.legend(fontsize=20, loc='lower right')

# Set font sizes
plt.xlabel(r'Contrast', fontsize=22)
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