#!/bin/bash
#SBATCH --job-name=data_process
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=30
##SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --mem=256G
#SBATCH --partition=All
#SBATCH --gpus-per-node=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --time=0-08:00:00
#SBATCH --open-mode=append
#SBATCH --container registry.gitlab.tech.orange/arthur.garon/hfsonar/snapshot:zero
#SBATCH --container-mounts ./:/workdir,/opt/marcel-c3/workdir/ygyd8053


pip install datasets==3.5

# HF_ENDPOINT="https://repos.tech.orange/artifactory/api/huggingfaceml/huggingface-proxy"
# export HF_ENDPOINT



# python data/speech/prepare_speech_ds.py \
#     --model_name models/pretrained/sonar_speech --encoder_name cointegrated/SONAR_200_text_encoder --data_path mozilla-foundation/common_voice_17_0 --output_path data/datasets/common_voice_17 --task_type embed --text_column sentence --num_proc $SLURM_CPUS_PER_TASK --lang en


python data/speech/prepare_eval.py \
    --model_name models/pretrained/sonar_speech --data_path google/fleurs --output_path data/datasets/fleurs --text_column raw_transcription --num_proc $SLURM_CPUS_PER_TASK --lang en_us
