#!/bin/bash
#SBATCH --job-name=data_process
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --partition=All
#SBATCH --gpus-per-node=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --time=0-08:00:00
#SBATCH --open-mode=append


pip install datasets==3.5 


python data/speech/prepare_speech_ds.py \
    --model_name models/pretrained/sonar_speech --encoder_name cointegrated/SONAR_200_text_encoder --data_path mozilla-foundation/common_voice_17_0 --output_path data/datasets/common_voice_17 --task_type embed --text_column sentence --num_proc $SLURM_CPUS_PER_TASK --lang en


python data/speech/prepare_eval.py \
    --model_name models/pretrained/sonar_speech --data_path google/fleurs --output_path data/datasets/fleurs --text_column raw_transcription --num_proc $SLURM_CPUS_PER_TASK --lang en_us
