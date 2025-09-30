# Open Sonar

Reproduction of the training pipeline "SONAR: Sentence-Level Multimodal and Language-Agnostic Representations" (https://arxiv.org/abs/2308.11466)

## Getting started



## Running jobs

Run in SLURM env:
sbatch scripts
if you need interactive use salloc

If you are note in the SLURM env, you can use sh scripts


## Run Script

This script is a **Slurm-friendly launcher** for training jobs with configurable options.  
It supports both **short flags** (`-c`) and **long flags** (`--config`) and automatically sets environment variables.

## Defaults

- **Program**: `run_translation_sd.py`  
- **Config**: `configs/training/text/sonar_sd.yaml`  
- **DeepSpeed stage**: `2`  
- **Accelerate config**: auto-determined from `DS_STAGE` unless overridden  

## Arguments

| Option | Long form | Description | Default |
|--------|-----------|-------------|---------|
| `-p`   | `--program`           | Python entrypoint to run | `run_translation_sd.py` |
| `-c`   | `--config`            | Training config YAML file | `configs/training/text/sonar_sd.yaml` |
| `-d`   | `--ds-stage`          | DeepSpeed stage (`-1` disables DS) | `2` |
| `-a`   | `--accelerate-config` | Explicit Accelerate config file | auto-computed |

## Logic for `ACCELERATE_CONFIG`

- If `--accelerate-config` is provided → use it directly  
- Else if `--ds-stage -1` → use `configs/accelerate/accelerate.yaml`  
- Else → use `configs/accelerate/accelerate_ds_<DS_STAGE>.yaml`  

## Examples

```bash
# Run with defaults
./run.sh

# Override config file
./run.sh --config configs/training/text/other.yaml

# Explicit program + config
./run.sh -p train.py -c myconfig.yaml

# Disable DeepSpeed, fall back to vanilla Accelerate
./run.sh --ds-stage -1

# Force a custom accelerate config
./run.sh --accelerate-config configs/accelerate/custom.yaml



