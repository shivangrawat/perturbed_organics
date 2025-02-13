import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import math
from perturbed_organics.spectrum_general import matrix_solution
from perturbed_organics.spectrum_general import sim_solution
import perturbed_organics.model.ORGaNICs_models as organics
import perturbed_organics.utils as utils
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import argparse
import os
from matplotlib import colors
from scipy.optimize import fsolve, curve_fit
from scipy import integrate
import warnings
import shutil
import json


# Import the arguments
parser = argparse.ArgumentParser(description="Sparse Matrix Stability Scan")

# arguments for the job array code
parser.add_argument(
    "--TASK_ID",
    type=int,
    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
    help="Task ID",
)
parser.add_argument("--NUM_TASKS", type=int, default=1, help="Number of tasks")

# parameters of the model
parser.add_argument("--MODEL_NAME", type=str, default="localized", help="Model name")
parser.add_argument("--MATRIX_TYPE", type=str, default="goe_symmetric", help="Random matrix type")
parser.add_argument("--initial_type", type=str, default="norm", help="Initial condition for simulation")
parser.add_argument("--data_save_loc", type=str, default="/vast/sr6364/perturbed_organics/data", help="Location to save data")
parser.add_argument("--extra_file_name", type=str, default="", help="Extra file name")
parser.add_argument("--N", type=int, default=100, help="Number of neurons")
parser.add_argument("--s", type=int, default=100, help="Sparsity")
parser.add_argument("--mu", type=float, default=0.0, help="Mean of the distribution")
parser.add_argument("--sigma", type=float, default=0.1, help="Sigma value")
parser.add_argument("--b0", type=float, default=0.5, help="Input gain for a")
parser.add_argument("--b1", type=float, default=0.5, help="Input gain for y")
parser.add_argument("--tauA", type=float, default=0.002, help="Time constant fot a")
parser.add_argument("--tauY", type=float, default=0.002, help="Time constant for y")

# parameters of simulation
parser.add_argument("--num_trials", type=int, default=10, help="Number of trials")
parser.add_argument("--num_delta", type=int, default=10, help="Number of delta steps")
parser.add_argument("--num_input", type=int, default=10, help="Number of input steps")
parser.add_argument(
    "--min_delta", type=float, default=0.0, help="Minimum value of the parameter delta"
)
parser.add_argument(
    "--max_delta", type=float, default=5.0, help="Maximum value of the parameter delta"
)
parser.add_argument(
    "--min_input", type=float, default=0.01, help="Minimum value of the parameter input"
)
parser.add_argument(
    "--max_input", type=float, default=5.0, help="Maximum value of the parameter input"
)

# parse the arguments
args = parser.parse_args()

# argument sof the job array
task_id = args.TASK_ID
num_tasks = args.NUM_TASKS

print(f"Task ID: {task_id}")

# arguments of the model parameters
model_name = args.MODEL_NAME
matrix_type = args.MATRIX_TYPE
data_save_loc = args.data_save_loc
extra_file_name = args.extra_file_name
initial_type = args.initial_type
N = args.N
s = args.s
mu = args.mu

# arguments of the simulation
num_delta = args.num_delta
num_input = args.num_input
num_trials = args.num_trials
min_input = args.min_input
max_input = args.max_input
min_delta = args.min_delta
max_delta = args.max_delta

# Define the scan parameters
delta_range = np.linspace(min_delta, max_delta, num_delta)
# input_range = np.linspace(min_input, max_input, num_input)
input_range = np.logspace(np.log10(min_input), np.log10(max_input), num_input)

# define the path of the folder to save the results in
folder_name = (
    model_name
    + "_"
    + matrix_type
    + "_N_{}_s_{}_mu_{}_num_delta_{}_num_input_{}_num_trials_{}_b0_{}_b1_{}_{}".format(
        N, s, mu, num_delta, num_input, num_trials, args.b0, args.b1, extra_file_name
    )
)
path = os.path.join(data_save_loc, folder_name)

# Save parameters only if task_id == 0 to avoid race conditions
if task_id == 0:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    # Save parameters to a JSON file in the folder
    params_to_save = {
        "model_name": model_name,
        "matrix_type": matrix_type,
        "N": N,
        "s": s,
        "mu": mu,
        "sigma": args.sigma,
        "b0": args.b0,
        "b1": args.b1,
        "tauA": args.tauA,
        "tauY": args.tauY,
        "num_trials": num_trials,
        "num_delta": num_delta,
        "num_input": num_input,
        "max_delta": max_delta,
        "max_input": max_input,
        "delta_range": delta_range.tolist(),
        "input_range": input_range.tolist(),
        "num_tasks": num_tasks,
    }
    param_file_path = os.path.join(path, "parameters.json")
    print("Saving the parameter file")
    with open(param_file_path, "w") as f:
        json.dump(params_to_save, f, indent=4)


device = torch.device("cpu")

# Define the parameters of the ORGaNICs
params = {"N_y": N, "N_a": N, "eta": 0.02, "noise_type": "additive"}
b0 = args.b0 * torch.ones(N)
b1 = args.b1 * torch.ones(N)
sigma = torch.tensor([args.sigma])
tauA = args.tauA + 0 * torch.abs(torch.randn(N) * 0.001)
tauY = args.tauY + 0 * torch.abs(torch.randn(N) * 0.001)
# Wyy = torch.eye(N)
Way = torch.ones(N, N)

# Create list of parameter combinations
param_combinations = [(i, j) for i in range(num_delta) for j in range(num_input)]

# Split parameter combinations among tasks
param_chunks = np.array_split(param_combinations, num_tasks)
my_params = param_chunks[task_id]


def run_trial(i, j, k, delta, input):
    # Initialize variables for the trial

    z = utils.make_input_drive(N=N, input_type=model_name, input_norm=input)

    Wyy = torch.eye(N) + utils.generate_matrix(
        N=N, matrix_type=matrix_type, s=s, mu=mu, delta=delta
    )

    # find the spectral radius of Wyy
    eigvals = torch.linalg.eigvals(Wyy)
    spectral_radius = torch.max(torch.abs(eigvals))

    # Instantiate the model
    model = organics.ORGaNICs2Dgeneral(
        params=params,
        b0=b0,
        b1=b1,
        sigma=sigma,
        tauA=tauA,
        tauY=tauY,
        Wyy=Wyy,
        Way=Way,
        z=z,
        initial_type=initial_type,
        run_jacobian=True,
    )

    y_s0, a_s0 = model.analytical_ss()
    y_s1, a_s1 = model.first_order_correction()

    if model.J is None:
        return (
            False,
            spectral_radius.to(torch.float16),
            y_s0.to(torch.float16),
            a_s0.to(torch.float16),
            None,
            None,
            y_s1.to(torch.float16),
            a_s1.to(torch.float16),
            None,
        )
    else:
        eigvals_J = torch.linalg.eigvals(model.J)
        return (
            True,
            spectral_radius.to(torch.float16),
            y_s0.to(torch.float16),
            a_s0.to(torch.float16),
            model.ss[0:N].to(torch.float16),
            model.ss[N : 2 * N].to(torch.float16),
            y_s1.to(torch.float16),
            a_s1.to(torch.float16),
            eigvals_J,
        )


def run_trial_and_collect(i, j, k):
    delta = delta_range[i]
    input = input_range[j]
    stable, spec_radius, y_s0, a_s0, y_s_actual, a_s_actual, y_s1, a_s1, eigvals_J = (
        run_trial(i, j, k, delta, input)
    )
    return (
        i,
        j,
        k,
        stable,
        spec_radius,
        y_s0,
        a_s0,
        y_s_actual,
        a_s_actual,
        y_s1,
        a_s1,
        eigvals_J,
    )


results = Parallel(n_jobs=-1, verbose=2)(
    delayed(run_trial_and_collect)(i, j, k)
    for i, j in my_params
    for k in range(num_trials)
)

# Collect and save partial results
bool_stable_task = torch.full(
    (num_delta, num_input, num_trials), fill_value=-1, dtype=torch.int8
)
spectral_radius_task = torch.full(
    (num_delta, num_input, num_trials), fill_value=float("nan"), dtype=torch.float16
)
norm_fixed_point_y_task = torch.full(
    (num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16
)
norm_fixed_point_a_task = torch.full(
    (num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16
)
actual_fixed_point_y_task = torch.full(
    (num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16
)
actual_fixed_point_a_task = torch.full(
    (num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16
)
first_order_perturb_y_task = torch.full(
    (num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16
)
first_order_perturb_a_task = torch.full(
    (num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16
)
eigvals_J_task = torch.full(
    (num_delta, num_input, num_trials, 2 * N),
    fill_value=float("nan"),
    dtype=torch.complex64,
)

# Update the results arrays
for res in results:
    (
        i,
        j,
        k,
        stable,
        spec_radius,
        y_s0,
        a_s0,
        y_s_actual,
        a_s_actual,
        y_s1,
        a_s1,
        eigvals_J,
    ) = res
    bool_stable_task[i, j, k] = int(stable)
    spectral_radius_task[i, j, k] = spec_radius
    norm_fixed_point_y_task[i, j, k] = y_s0
    norm_fixed_point_a_task[i, j, k] = a_s0
    first_order_perturb_y_task[i, j, k] = y_s1
    first_order_perturb_a_task[i, j, k] = a_s1

    if y_s_actual is not None:
        actual_fixed_point_y_task[i, j, k] = y_s_actual
        actual_fixed_point_a_task[i, j, k] = a_s_actual
        eigvals_J_task[i, j, k] = eigvals_J

# delete results to save memory
del results

# Save partial results
torch.save(bool_stable_task, os.path.join(path, f"bool_stable_task_{task_id}.pt"))
torch.save(
    spectral_radius_task, os.path.join(path, f"spectral_radius_task_{task_id}.pt")
)
torch.save(
    norm_fixed_point_y_task,
    os.path.join(path, f"norm_fixed_point_y_task_{task_id}.pt"),
)
torch.save(
    norm_fixed_point_a_task,
    os.path.join(path, f"norm_fixed_point_a_task_{task_id}.pt"),
)
torch.save(
    actual_fixed_point_y_task,
    os.path.join(path, f"actual_fixed_point_y_task_{task_id}.pt"),
)
torch.save(
    actual_fixed_point_a_task,
    os.path.join(path, f"actual_fixed_point_a_task_{task_id}.pt"),
)
torch.save(
    first_order_perturb_y_task,
    os.path.join(path, f"first_order_perturb_y_task_{task_id}.pt"),
)
torch.save(
    first_order_perturb_a_task,
    os.path.join(path, f"first_order_perturb_a_task_{task_id}.pt"),
)
torch.save(
    eigvals_J_task,
    os.path.join(path, f"eigvals_J_task_{task_id}.pt"),
)
