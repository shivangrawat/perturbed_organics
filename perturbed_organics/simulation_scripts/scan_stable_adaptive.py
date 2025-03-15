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

# Define the torch data type and set default dtype
torch_data_type = torch.float64
torch.set_default_dtype(torch_data_type)

# Import the arguments
parser = argparse.ArgumentParser(description="Sparse Matrix Stability Scan")

# Arguments for the job array code
parser.add_argument(
    "--TASK_ID",
    type=int,
    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
    help="Task ID",
)
parser.add_argument("--NUM_TASKS", type=int, default=1, help="Number of tasks")

# Parameters of the model
parser.add_argument("--MODEL_NAME", type=str, default="delocalized", help="Model name")
parser.add_argument("--MATRIX_TYPE", type=str, default="goe_symmetric", help="Random matrix type")
parser.add_argument("--initial_type", type=str, default="norm", help="Initial condition for simulation")
parser.add_argument("--input_scale", type=str, default="log-scale", help="How to sample the input strength")
parser.add_argument("--data_save_loc", type=str, default="/vast/sr6364/perturbed_organics/data", help="Location to save data")
parser.add_argument("--extra_file_name", type=str, default="", help="Extra file name")
parser.add_argument("--N", type=int, default=100, help="Number of neurons")
parser.add_argument("--s", type=int, default=100, help="Sparsity")
parser.add_argument("--mu", type=float, default=0.0, help="Mean of the distribution")
parser.add_argument("--sigma", type=float, default=0.1, help="Sigma value")
parser.add_argument("--b0", type=float, default=0.5, help="Input gain for a")
parser.add_argument("--b1", type=float, default=0.5, help="Input gain for y")
parser.add_argument("--tauA", type=float, default=0.002, help="Time constant for a")
parser.add_argument("--tauY", type=float, default=0.002, help="Time constant for y")

# Parameters of simulation
parser.add_argument("--num_trials", type=int, default=10, help="Number of trials")
parser.add_argument("--num_delta", type=int, default=10, help="Number of delta steps")
parser.add_argument("--num_input", type=int, default=10, help="Number of input steps")
parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum value of the parameter delta")
parser.add_argument("--max_delta", type=float, default=5.0, help="Maximum value of the parameter delta")
parser.add_argument("--min_input", type=float, default=0.01, help="Minimum value of the parameter input")
parser.add_argument("--max_input", type=float, default=5.0, help="Maximum value of the parameter input")

# Parse the arguments
args = parser.parse_args()

# Arguments for the job array
task_id = args.TASK_ID
num_tasks = args.NUM_TASKS

print(f"Task ID: {task_id}")

# Arguments of the model parameters
model_name = args.MODEL_NAME
matrix_type = args.MATRIX_TYPE
input_scale = args.input_scale
data_save_loc = args.data_save_loc
extra_file_name = args.extra_file_name
initial_type = args.initial_type
N = args.N
s = args.s
mu = args.mu

# Arguments of the simulation
num_delta = args.num_delta
num_input = args.num_input
num_trials = args.num_trials
min_input = args.min_input
max_input = args.max_input
min_delta = args.min_delta
max_delta = args.max_delta

# Define the scan parameters
delta_range = np.linspace(min_delta, max_delta, num_delta)
if input_scale == "log-scale":
    input_range = np.logspace(np.log10(min_input), np.log10(max_input), num_input)
else:
    input_range = np.linspace(min_input, max_input, num_input)

# Define the folder path to save results
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
    params_to_save = {
        "model_name": model_name,
        "matrix_type": matrix_type,
        "initial_type": initial_type,
        "input_scale": input_scale,
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
b0 = args.b0 * torch.ones(N, dtype=torch_data_type)
b1 = args.b1 * torch.ones(N, dtype=torch_data_type)
sigma = torch.tensor([args.sigma], dtype=torch_data_type)
tauA = args.tauA + 0 * torch.abs(torch.randn(N, dtype=torch_data_type) * 0.001)
tauY = args.tauY + 0 * torch.abs(torch.randn(N, dtype=torch_data_type) * 0.001)
Way = torch.ones(N, N, dtype=torch_data_type)

# Create a full list of job indices covering all (delta, input, trial) combinations
jobs = [(i, j, k) for i in range(num_delta)
                 for j in range(num_input)
                 for k in range(num_trials)]
# Split the jobs among tasks
jobs_chunks = np.array_split(jobs, num_tasks)
my_jobs = jobs_chunks[task_id]  # Each task processes its own subset


def run_trial(i, j, k, delta, input):
    # Initialize variables for the trial

    # Pass the torch_data_type to the util function calls
    z = utils.make_input_drive(N=N, input_type=model_name, input_norm=input, dtype=torch_data_type)
    Wyy = torch.eye(N, dtype=torch_data_type) + utils.generate_matrix(
        N=N, matrix_type=matrix_type, s=s, mu=mu, delta=delta, dtype=torch_data_type
    )

    # Find the spectral radius of Wyy
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
        run_jacobian=False,
    )

    y_s0, a_s0 = model.analytical_ss()
    y_s1, a_s1 = model.first_order_correction()

    # Run adaptive simulation scheme and find the dynamics

    # Define the time vector
    tau_min = min(torch.min(tauA), torch.min(tauY))
    tau_max = max(torch.max(tauA), torch.max(tauY))
    chunk_time = 100 * tau_max  # Simulate in chunks of 100 * tau_max
    dt = 0.05 * tau_min  # Defines the timestep of simulation
    points = int(chunk_time / dt)  # Number of points per chunk
    t_chunk = torch.linspace(0, chunk_time, points, dtype=torch_data_type)

    max_loops = 20
    condition = 0  # 0: not met, 1: diverging, 2: fixed point, 3: periodic

    # Define the model for simulation
    y0 = model.inital_conditions(initial_type=initial_type)
    sim_obj = sim_solution(model)

    for loop in range(max_loops):
        traj_segment = sim_obj.simulate(t_chunk, y0=y0)
        y0 = traj_segment[-1, :]

        if utils.is_diverging(traj_segment):
            condition = 1
            break
        elif utils.is_fixed_point(traj_segment):
            condition = 2
            break
        elif utils.is_periodic(traj_segment[:, 0].detach().numpy()):
            condition = 3
            break

    if condition == 2:
        # Find the jacobian and its eigenvalue
        ss = traj_segment[-1, :]
        J, _ = model.jacobian_autograd(ss=ss)
        eigvals_J = torch.linalg.eigvals(J)
        return (
            condition,
            spectral_radius.to(torch.float16),
            y_s0.to(torch.float16),
            a_s0.to(torch.float16),
            ss[0:N].to(torch.float16),
            ss[N: 2 * N].to(torch.float16),
            y_s1.to(torch.float16),
            a_s1.to(torch.float16),
            eigvals_J,
        )
    else:
        return (
            condition,
            spectral_radius.to(torch.float16),
            y_s0.to(torch.float16),
            a_s0.to(torch.float16),
            None,
            None,
            y_s1.to(torch.float16),
            a_s1.to(torch.float16),
            None,
        )


def run_trial_and_collect(i, j, k):
    delta = delta_range[i]
    input = input_range[j]
    condition, spec_radius, y_s0, a_s0, y_s_actual, a_s_actual, y_s1, a_s1, eigvals_J = run_trial(i, j, k, delta, input)
    return (
        i,
        j,
        k,
        condition,
        spec_radius,
        y_s0,
        a_s0,
        y_s_actual,
        a_s_actual,
        y_s1,
        a_s1,
        eigvals_J,
    )

# Run the jobs assigned to this task
results = Parallel(n_jobs=-1, verbose=2)(
    delayed(run_trial_and_collect)(i, j, k) for i, j, k in my_jobs
)


# Collect and save partial results
condition_task = torch.full((num_delta, num_input, num_trials), fill_value=-1, dtype=torch.int8)
spectral_radius_task = torch.full((num_delta, num_input, num_trials), fill_value=float("nan"), dtype=torch.float16)
norm_fixed_point_y_task = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
norm_fixed_point_a_task = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
actual_fixed_point_y_task = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
actual_fixed_point_a_task = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
first_order_perturb_y_task = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
first_order_perturb_a_task = torch.full((num_delta, num_input, num_trials, N), fill_value=float("nan"), dtype=torch.float16)
eigvals_J_task = torch.full((num_delta, num_input, num_trials, 2 * N), fill_value=float("nan"), dtype=torch.complex64)

# Update the results arrays
for res in results:
    (i, j, k, condition, spec_radius, y_s0, a_s0, y_s_actual, a_s_actual, y_s1, a_s1, eigvals_J) = res
    condition_task[i, j, k] = condition
    spectral_radius_task[i, j, k] = spec_radius
    norm_fixed_point_y_task[i, j, k] = y_s0
    norm_fixed_point_a_task[i, j, k] = a_s0
    first_order_perturb_y_task[i, j, k] = y_s1
    first_order_perturb_a_task[i, j, k] = a_s1

    if y_s_actual is not None:
        actual_fixed_point_y_task[i, j, k] = y_s_actual
        actual_fixed_point_a_task[i, j, k] = a_s_actual
        eigvals_J_task[i, j, k] = eigvals_J

# Delete results to save memory
del results

# Save partial results
torch.save(condition_task, os.path.join(path, f"condition_task_{task_id}.pt"))
torch.save(spectral_radius_task, os.path.join(path, f"spectral_radius_task_{task_id}.pt"))
torch.save(norm_fixed_point_y_task, os.path.join(path, f"norm_fixed_point_y_task_{task_id}.pt"))
torch.save(norm_fixed_point_a_task, os.path.join(path, f"norm_fixed_point_a_task_{task_id}.pt"))
torch.save(actual_fixed_point_y_task, os.path.join(path, f"actual_fixed_point_y_task_{task_id}.pt"))
torch.save(actual_fixed_point_a_task, os.path.join(path, f"actual_fixed_point_a_task_{task_id}.pt"))
torch.save(first_order_perturb_y_task, os.path.join(path, f"first_order_perturb_y_task_{task_id}.pt"))
torch.save(first_order_perturb_a_task, os.path.join(path, f"first_order_perturb_a_task_{task_id}.pt"))
torch.save(eigvals_J_task, os.path.join(path, f"eigvals_J_task_{task_id}.pt"))
