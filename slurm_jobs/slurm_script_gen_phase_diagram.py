#!/usr/bin/env python3
import argparse
import os

# model_type = ""
model_type = "_rectified"
# model_type = "_rectified_recurrence"


# Define default parameters
default_params = {
    # "MODEL_NAME": "localized",
    'MODEL_NAME': 'delocalized',
    # 'MODEL_NAME': 'random',
    # 'MODEL_NAME': 'gaussian',
    # 'MATRIX_TYPE': 'goe',
    "MATRIX_TYPE": "goe_symmetric",
    # 'MATRIX_TYPE': 'power_law',
    # "initial_type": "norm",
    "initial_type": "first_order",
    "delta_scale": "log-scale",
    # "delta_scale": "linear",
    "N": 100,
    "s": 100,
    "mu": 1.0,
    "sigma": 0.1,
    "b0": 1.0,
    "b1": 1.0,
    "tauA": 0.002,
    "tauY": 0.002,
    "num_trials": 100,
    "num_delta": 200,
    "num_input": 200,
    "min_delta": 0.01,
    "max_delta": 10.0,
    "min_input": 0.01,
    "max_input": 1.0,
    "NUM_TASKS": 100,
    "JOB_NAME": "param_scan",
    "TIME": "15:00:00",
    "CPUS": 4,
    "MEMORY": "16GB",
    "EMAIL": "sr6364@nyu.edu",
    "SCRIPT_NAME": f"scan_stable_adaptive{model_type}.py",
}

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate SLURM job submission script.")

# Add arguments for parameters you may want to change
for param in default_params:
    if isinstance(default_params[param], bool):
        parser.add_argument(
            f"--{param}", action="store_true", default=default_params[param]
        )
    else:
        parser.add_argument(
            f"--{param}",
            type=type(default_params[param]),
            default=default_params[param],
        )

args = parser.parse_args()

# Create directory based on MODEL_NAME and matrix type if it doesn't exist
output_dir = args.MODEL_NAME
os.makedirs(output_dir, exist_ok=True)


##### Change things here for a better filename
data_save_loc = f"/scratch/sr6364/perturbed_organics/data/adaptive_phase_diagram_100_large_delta{model_type}/{args.MODEL_NAME}"
# extra_file_name = f"delta_{args.min_delta}"
extra_file_name = f"phase_diagram_{args.delta_scale}{model_type}"

# Generate the filename based on selected parameters
filename = os.path.join(
    output_dir,
    f"job_{args.MODEL_NAME}_N{args.N}_s{args.s}_mu{args.mu}_{extra_file_name}.sh",
)


# Now generate the job submission script
job_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time={args.TIME}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.CPUS}
#SBATCH --job-name={args.JOB_NAME}
#SBATCH --mem={args.MEMORY}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={args.EMAIL}
#SBATCH --output=job.%A_%a.out
#SBATCH --array=0-{args.NUM_TASKS - 1}

singularity exec --overlay /scratch/sr6364/overlay-files/overlay-50G-10M.ext3:ro \\
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c \\
'source /ext3/env.sh; conda activate feed-r-conda; \\
cd /home/sr6364/python_scripts/perturbed_organics/perturbed_organics/simulation_scripts; \\
python {args.SCRIPT_NAME} \\
    --MODEL_NAME {args.MODEL_NAME} \\
    --MATRIX_TYPE {args.MATRIX_TYPE} \\
    --initial_type {args.initial_type} \\
    --delta_scale {args.delta_scale} \\
    --N {args.N} \\
    --s {args.s} \\
    --mu {args.mu} \\
    --sigma {args.sigma} \\
    --b0 {args.b0} \\
    --b1 {args.b1} \\
    --tauA {args.tauA} \\
    --tauY {args.tauY} \\
    --num_trials {args.num_trials} \\
    --num_delta {args.num_delta} \\
    --num_input {args.num_input} \\
    --min_delta {args.min_delta} \\
    --max_delta {args.max_delta} \\
    --min_input {args.min_input} \\
    --max_input {args.max_input} \\
    --data_save_loc {data_save_loc} \\
    --extra_file_name {extra_file_name} \\
    --TASK_ID ${{SLURM_ARRAY_TASK_ID}} \\
    --NUM_TASKS {args.NUM_TASKS} '
"""


# Write the job script to a file
with open(filename, "w") as f:
    f.write(job_script)


# Now we create the combine script
# define the folder name based on the parameters
folder_loc = data_save_loc
folder_name = f"{args.MODEL_NAME}_{args.MATRIX_TYPE}_N_{args.N}_s_{args.s}_mu_{args.mu}_num_delta_{args.num_delta}_num_input_{args.num_input}_num_trials_{args.num_trials}_b0_{args.b0}_b1_{args.b1}_{extra_file_name}"

combine_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=combine_plot
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr6364@nyu.edu
#SBATCH --output=job.%j.out

singularity exec --overlay /scratch/sr6364/overlay-files/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c \
'source /ext3/env.sh; conda activate feed-r-conda; cd /home/sr6364/python_scripts/perturbed_organics/perturbed_organics/simulation_scripts; python combine_adaptive.py \\
--folder_loc {folder_loc} \\
--folder_name {folder_name} '
"""

filename = os.path.join(
    output_dir,
    f"combine_plot_{args.MODEL_NAME}_{args.MATRIX_TYPE}_N{args.N}_s{args.s}_mu{args.mu}_{extra_file_name}.sh",
)

# save the file
# Write the job script to a file
with open(filename, "w") as f:
    f.write(combine_script)
