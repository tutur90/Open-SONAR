#!/bin/bash
#SBATCH --job-name=sonar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
##SBATCH --cpus-per-task=5
##SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --mem=256G
#SBATCH --partition=multigpu,All
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/%x/%j.out
#SBATCH --err=logs/%x/%j.err
#SBATCH --time=3-00:00:00
#SBATCH --constraint=gpu_mem_80
#SBATCH --open-mode=append
#SBATCH --cpus-per-gpu=4


echo "[$SLURM_JOB_ID] Started '$(date)' on '$(hostname)' with args: '$*'"

# Default values
PROGRAM="run_translation.py"
CONFIG="configs/training/text/sonar.yaml"
DS_STAGE=2

# Parse args (GNU getopt required)
OPTS=$(getopt -o p:c:d:a:e: -l program:,config:,ds-stage:,accelerate-config:,conda-env: -- "$@")
eval set -- "$OPTS"

while true; do
  case "$1" in
    -p|--program) PROGRAM="$2"; shift 2 ;;
    -c|--config)  CONFIG="$2"; shift 2 ;;
    -d|--ds-stage) DS_STAGE="$2"; shift 2 ;;
    -a|--accelerate-config) ACCELERATE_CONFIG="$2"; shift 2 ;;
    -e|--conda-env) CONDA_ENV="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

# Compute accelerate config
if [ "$DS_STAGE" == "-1" ] && [ -z "$ACCELERATE_CONFIG" ]; then
    ACCELERATE_CONFIG="configs/accelerate/accelerate.yaml"
elif [ -z "$ACCELERATE_CONFIG" ]; then
    ACCELERATE_CONFIG="configs/accelerate/accelerate_ds_${DS_STAGE}.yaml"
fi

# Export so children see them
export PROGRAM CONFIG ACCELERATE_CONFIG CONDA_ENV

# Print summary
echo "PROGRAM=$PROGRAM"
echo "CONFIG=$CONFIG"
echo "DS_STAGE=$DS_STAGE"
echo "ACCELERATE_CONFIG=$ACCELERATE_CONFIG"



# Docker configuration
CONTAINER_IMAGE=...
CONTAINER_MOUNTS=./:/workdir


# GPUS and distributed training configuration
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
WORLD_SIZE=$(($SLURM_NNODES * GPUS_PER_NODE))
export WORLD_SIZE
export NNODES=$SLURM_NNODES


# Network configuration
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=$(getent hosts $MASTER_NODE | awk '{print $1}')
export MASTER_ADDR


if [ -z "$CONDA_ENV" ]; then
    docker run $CONTAINER_IMAGE scripts/run.sh
else
    scripts/run.sh

fi

echo "END TIME: $(date)"