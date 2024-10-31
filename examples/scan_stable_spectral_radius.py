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
parser.add_argument('--MODEL_NAME', type=str, default="scan", help='Model name')
parser.add_argument('--N', type=int, default=100, help="Number of neurons")
parser.add_argument('--c', type=int, default=100, help="Sparsity")
parser.add_argument('--num_delta', type=int, default=10, help="Number of delta steps")
parser.add_argument('--num_gamma', type=int, default=10, help="Number of gamma steps")
parser.add_argument('--num_trials', type=int, default=10, help="Number of trials")
parser.add_argument('--mu', type=float, default=0.0, help="Mean of the distribution")
args = parser.parse_args()

# Assign the arguments to the variables
model_name = args.MODEL_NAME
N = args.N
c = args.c
mu = args.mu
num_delta = args.num_delta
num_gamma = args.num_gamma
num_trials = args.num_trials

device = torch.device("cpu")

# Define the parameters of the ORGaNICs
params = {
    'N_y': N,
    'N_a': N,
    'eta': 0.02,
    'noise_type': 'additive'
}
b0 = 0.8 * torch.ones(N)
b1 = 0.4 * torch.ones(N)
sigma = torch.tensor([1.0])
tauA = 0.02 + 0 * torch.abs(torch.randn(N) * 0.001)
tauY = 0.002 + 0 * torch.abs(torch.randn(N) * 0.001)
# Wyy = torch.eye(N)
Way = torch.ones(N, N)

def sample_sparse_matrix(N, c, delta, mu):
    mask = torch.bernoulli(torch.full((N, N), c / N))
    values = torch.eye(N) + torch.normal(mu, 1.0, (N, N))
    sparse_matrix = values * mask
    eigvals = torch.linalg.eigvals(sparse_matrix)
    spectral_radius = torch.max(torch.abs(eigvals))

    scaling_factor = delta / spectral_radius
    sparse_matrix *= scaling_factor
    return sparse_matrix


# Define the scan parameters
delta_range = np.linspace(0, 10, num_delta)
gamma_range = np.linspace(0.01, 3, num_gamma)

# define the quantities that we'll calculate
bool_stable = torch.zeros((num_delta, num_gamma, num_trials), dtype=torch.bool)

def run_trial(i, j, k, delta, gamma):
    # Initialize variables for the trial

    # # make input delocalized
    z = torch.ones(N)
    z = z / torch.norm(z) * gamma

    # make input one hot
    # z = torch.zeros(N)
    # z[0] = gamma

    Wyy = sample_sparse_matrix(N, c, delta, mu)

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
        return False
    else:
        return True

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_trial)(i, j, k, delta, gamma)
    for i, delta in enumerate(delta_range)
    for j, gamma in enumerate(gamma_range)
    for k in range(num_trials)
)

for idx, (i, j, k) in enumerate(
    (i, j, k) for i in range(num_delta) for j in range(num_gamma) for k in range(num_trials)
):
    bool_stable[i, j, k] = results[idx]

# create a folder in ..data/ to save the results
folder_name = model_name + '_N_{}_c_{}_mu_{}_num_delta_{}_num_gamma_{}_num_trials_{}'.format(N, c, mu, num_delta, num_gamma, num_trials)
path = os.path.join('..', 'data', folder_name)

if not os.path.exists(path):
    os.makedirs(path)

torch.save(bool_stable, os.path.join(path, 'bool_stable.pt'))


# plot the circuit
percent_stable = bool_stable.float().mean(dim=2) * 100

# Plot the heatmap
plt.figure(figsize=(12, 10))
plt.imshow(percent_stable, extent=[gamma_range.min(), gamma_range.max(), delta_range.min(), delta_range.max()],
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