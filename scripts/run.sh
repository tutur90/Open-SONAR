#!/bin/sh

# Conda or Docker
if [ -z "$CONDA_ENV" ]; then
    echo "Using docker container"
else
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    echo "Using conda environment: $CONDA_ENV"
fi

# Defaults
: "${NNODES:=1}"
: "${SLURM_NODEID:=0}"
: "${MASTER_ADDR:=localhost}"
: "${WORLD_SIZE:=1}"


MASTER_PORT="${SLURM_STEP_RESV_PORTS%%-*}"

accelerate launch \
    --config_file  $ACCELERATE_CONFIG \
    --num_machines $NNODES \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    "$PROGRAM" "$CONFIG"
