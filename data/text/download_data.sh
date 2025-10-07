#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=All
#SBATCH --cpus-per-node=64
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append


# Set download directory
SCRIPT_DIR="data/text/downloading"
DOWNLOAD_DIRECTORY="data/.cache/downloads"

LOG_LEVEL="INFO"


NUM_PROC=$SLURM_CPUS_PER_TASK
echo "Using $NUM_PROC processes for downloading and processing datasets."



echo "Downloading flores200 dataset..."
python "$SCRIPT_DIR/flores200.py" 


# Step 1: Download Public Datasets
echo "Downloading public datasets..."
python "$SCRIPT_DIR/download_parallel_corpora.py" --directory "$DOWNLOAD_DIRECTORY/primary"
python "$SCRIPT_DIR/process_corpora.py" \
    --input_directory "$DOWNLOAD_DIRECTORY/primary" \
    --output_directory "$DOWNLOAD_DIRECTORY/formate/primary" \
    --logfile "$SCRIPT_DIR/datasets/microshards/primary/download.log" \
    --loglevel "$LOG_LEVEL" \



# Step 2: Download mined datasets
echo "Downloading mined datasets..."
python "$SCRIPT_DIR/download_mined.py" \
    --directory "$TRAIN_DIRECTORY/mined/parquet" \
    --logfile "$SCRIPT_DIR/datasets/mined/download.log" \
    --loglevel "$LOG_LEVEL" \
    --num_proc $NUM_PROC \
    --dataset "$SCRIPT_DIR/datasets/microshards/mined" \



echo "Downloading seed datasets..."
# git clone https://github.com/facebookresearch/flores.git
python "$SCRIPT_DIR/download_oldi.py" \
    --output_directory "$DOWNLOAD_DIRECTORY/formated" \
    --logfile "$DOWNLOAD_DIRECTORY/seed.lg" \
    --loglevel "$LOG_LEVEL" \



python "$SCRIPT_DIR/download_oldi.py" \
    --output_directory "$DOWNLOAD_DIRECTORY/formated" \
    --logfile "$DOWNLOAD_DIRECTORY/seed.lg" \
    --loglevel "$LOG_LEVEL" \
    --backtranslate_path "$DOWNLOAD_DIRECTORY/seed/predictions.csv" 



echo "Dataset downloaded!"
