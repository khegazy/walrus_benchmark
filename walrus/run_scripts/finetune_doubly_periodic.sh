#!/bin/bash -l
#SBATCH --time=23:00:00
#SBATCH -C gpu&hbm80g
#SBATCH -A m4790
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH -J finetune_doubly_periodic
#SBATCH --output=finetune_doubly_periodic_%j.log
#SBATCH -q premium

export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

module load conda
conda activate walrus

cd /global/u2/k/khegazy/projects/pde/walrus/walrus

srun python -u $(which torchrun) \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):29500 \
    train.py --config-name=finetune_doubly_periodic
