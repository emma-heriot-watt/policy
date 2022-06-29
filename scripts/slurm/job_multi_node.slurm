#!/bin/bash
#
#SBATCH --job-name=emma_pretraining
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --account=ec202

# Load the required modules
module load nvidia/nvhpc
# The following loads the central pytorch environment. Instead you need to initialise your environment:
module load pytorch/1.11.0-gpu

# Add your wandb credentials
export WANDB_API_KEY=fakekey3
export WANDB_CONFIG_DIR=/work/ec202/ec202/shared/.config/wadnb
export WANDB_CACHE_DIR=/work/ec202/ec202/shared/.cache/wadnb

# Set the following environment variables when running on multiple nodes
# We assume 4 GPUs per node
export SLURM_NTASKS=$((4 * SLURM_NNODES))
export SLURM_NTASKS_PER_NODE=$(expr ${SLURM_NTASKS} \/ ${SLURM_NNODES})
export SLURM_TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE}(x${SLURM_NNODES})"

srun python run.py trainer=ddp trainer.num_nodes=$SLURM_NNODES trainer.devices=$SLURM_TASKS_PER_NODE
