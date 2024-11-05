#!/usr/bin/env python3

import argparse
import os

# Define default parameters
default_params = {
    'MODEL_NAME': 'localized',
    # 'MODEL_NAME': 'delocalized',
    # 'MODEL_NAME': 'random',
    # 'MODEL_NAME': 'gaussian',
    'N': 100,
    'c': 3,
    'mu': 0.0,
    'sigma': 0.1,
    'b0': 0.5,
    'b1': 0.5,
    'tauA': 0.002,
    'tauY': 0.002,
    'num_trials': 10,
    'num_delta': 50,
    'num_input': 50,
    'max_delta': 5.0,
    'max_input': 5.0,
    'NUM_TASKS': 10,
    'JOB_NAME': 'param_scan',
    'TIME': '0:05:00',
    'CPUS': 32,
    'MEMORY': '32GB',
    'EMAIL': 'sr6364@nyu.edu',
    'SCRIPT_NAME': 'scan_stable.py'
}

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate SLURM job submission script.')

# Add arguments for parameters you may want to change
for param in default_params:
    if isinstance(default_params[param], bool):
        parser.add_argument(f'--{param}', action='store_true', default=default_params[param])
    else:
        parser.add_argument(f'--{param}', type=type(default_params[param]), default=default_params[param])

args = parser.parse_args()

# Create directory based on MODEL_NAME if it doesn't exist
output_dir = args.MODEL_NAME
os.makedirs(output_dir, exist_ok=True)

# Generate the filename based on selected parameters
filename = os.path.join(output_dir, f"job_submission_{args.MODEL_NAME}_N{args.N}_c{args.c}_mu{args.mu}.sh")


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
cd /home/sr6364/python_scripts/perturbed_organics/examples; \\
python {args.SCRIPT_NAME} \\
    --MODEL_NAME {args.MODEL_NAME} \\
    --N {args.N} \\
    --c {args.c} \\
    --mu {args.mu} \\
    --sigma {args.sigma} \\
    --b0 {args.b0} \\
    --b1 {args.b1} \\
    --tauA {args.tauA} \\
    --tauY {args.tauY} \\
    --num_trials {args.num_trials} \\
    --num_delta {args.num_delta} \\
    --num_input {args.num_input} \\
    --max_delta {args.max_delta} \\
    --max_input {args.max_input} \\
    --TASK_ID ${{SLURM_ARRAY_TASK_ID}} \\
    --NUM_TASKS {args.NUM_TASKS}'
"""

# 

# Write the job script to a file
with open(filename, 'w') as f:
    f.write(job_script)

