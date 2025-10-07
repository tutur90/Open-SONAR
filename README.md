# Open Sonar

Reproduction of the training pipeline for "SONAR: Sentence-Level Multimodal and Language-Agnostic Representations" (https://arxiv.org/abs/2308.11466)

## Getting Started

### Environment Setup

Create a conda environment:

```bash
conda create -n sonar python=3.11 -y
conda activate sonar
```

Install requirements:

```bash
pip install -r requirements.txt
```

Alternatively, use Docker:

```bash
docker build -t sonar .
```

### SONAR Text

#### Data Preparation

1. **Download datasets**: Download the Primary and Mined datasets from [NLLB](https://arxiv.org/abs/2207.04672). The download sources can be found in the scripts. To download all datasets, run:

```bash
bash data/download_datasets.sh
```

2. **Backtranslation** (recommended if languages are missing in the initial data):

```bash
bash data/backtranslate.sh
```

You will need to adapt `scripts/text/backtranslate.sbatch` and `configs/generation/backtranslate.yaml` to your specific needs, such as adjusting resource allocations or changing the backtranslation model.

3. **Preprocess the data**:

```bash
bash data/preprocess.sh
```

#### Fine-tuning SONAR

To fine-tune a pre-trained SONAR model, use the `configs/training/text/finetune.yaml` configuration file. This file is set up to load a pre-trained model and continue training on your specified dataset.

Run the training script with the following command:

```bash
sbatch scripts/text/run.sbatch --config configs/training/text/finetune.yaml
```

#### Training from Scratch

1. **Create your model**:

```bash
python open_sonar/text/models/create_model.py --model_id facebook/nllb-200-1.3B --model_output_path open_sonar/text/models/pretrained/nllb_1.3B
```

2. **Train the model**: To train a SONAR model from scratch, use the `configs/training/text/sonar.yaml` configuration file. This file is configured to initialize a new model and train it on your dataset.

```bash
sbatch scripts/text/run.sbatch --config configs/training/text/sonar.yaml
```

### SONAR Speech

**Note**: The speech component is still under development and may not work as expected. Hyperparameters are not fully tuned, and the code has not been fully tested.

#### Data Preparation

Download and preprocess the datasets (CommonVoice, BABEL, FLEURS, MLS). The download sources can be found in the scripts. To download all datasets, run:

```bash
bash data/speech/prepare.sh
```

#### Training

1. **Create your model**:

```bash
python open_sonar/speech/models/create_model.py --encoder_id facebook/w2v-bert-2.0 --model_id facebook/wav2vec2 --model_output_path open_sonar/speech/models/pretrained/sonar_speech
```

2. **Train the model**: To train a SONAR model for speech, use the `configs/training/speech/sonar.yaml` configuration file:

```bash
sbatch scripts/speech/run.sbatch --config configs/training/speech/sonar.yaml
```

## Scripts

### Running Jobs

**In a SLURM environment**:
- Use `sbatch scripts/<script_name>` to submit jobs
- Use `salloc` for interactive sessions

**Outside a SLURM environment**:
- You can run the scripts directly using `sh scripts/<script_name>`

### Run Script

This script is a **Slurm-friendly launcher** for training jobs with configurable options. It supports both **short flags** (`-c`) and **long flags** (`--config`) and automatically sets environment variables.

#### Defaults

- **Program**: `run_translation.py`
- **Config**: `configs/training/text/sonar.yaml`
- **DeepSpeed stage**: `2`
- **Accelerate config**: Auto-determined from `DS_STAGE` unless overridden

#### Arguments

| Option | Long Form | Description | Default |
|--------|-----------|-------------|---------|
| `-p` | `--program` | Python entrypoint to run | `run_translation_sd.py` |
| `-c` | `--config` | Training config YAML file | `configs/training/text/sonar_sd.yaml` |
| `-d` | `--ds-stage` | DeepSpeed stage (`-1` disables DeepSpeed) | `2` |
| `-a` | `--accelerate-config` | Explicit Accelerate config file | Auto-computed |

#### Logic for `ACCELERATE_CONFIG`

- If `--accelerate-config` is provided → use it directly
- Else if `--ds-stage -1` → use `configs/accelerate/accelerate.yaml`
- Else → use `configs/accelerate/accelerate_ds_<DS_STAGE>.yaml`

#### Examples

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
```