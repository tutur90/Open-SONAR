#!/bin/bash
#SBATCH --job-name=data_process
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64
##SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --mem=256G
#SBATCH --partition=All
#SBATCH --gpus-per-node=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --container registry.gitlab.tech.orange/arthur.garon/hfsonar/snapshot:zero
#SBATCH --container-mounts ./:/workdir,/opt/marcel-c3/workdir/ygyd8053

pip install polars
pip install polars-u64-idx


echo "Starting data processing with $SLURM_CPUS_PER_TASK CPUs"

# HF_ENDPOINT="https://repos.tech.orange/artifactory/api/huggingfaceml/huggingface-proxy"
# export HF_ENDPOINT



CACHE_DIR="data/.cache"

SCRIPTS_DIR="data/text"

################
# LID Thresholds
################

python $SCRIPTS_DIR/utils/get_column_stats.py \
    --input_files $CACHE_DIR/downloads/formated --output_path $CACHE_DIR/processing/lid_thresholds --column lid

##################
# Length Filtering
##################

# Processed separately but can be done at once with: --input_files data/downloads/formated


python $SCRIPTS_DIR/processing/length_filter.py --dataset_path "datasets/flores200/flores_200_dev.parquet" \
    --input_files $CACHE_DIR/downloads/formated/mined --output_path $CACHE_DIR/processing/length_filtered/mined --min_sentence_length 15 --max_length_ratio 4.0

python $SCRIPTS_DIR/processing/length_filter.py --dataset_path "datasets/flores200/flores_200_dev.parquet" \
    --input_files $CACHE_DIR/downloads/formated/primary --output_path $CACHE_DIR/processing/length_filtered/primary --min_sentence_length 15 --max_length_ratio 4.0

python $SCRIPTS_DIR/processing/length_filter.py --dataset_path "datasets/flores200/flores_200_dev.parquet" \
    --input_files $CACHE_DIR/downloads/formated/seed --output_path $CACHE_DIR/processing/length_filtered/seed --min_sentence_length 15 --max_length_ratio 4.0


# #################
# # LID Filtering
# #################

# #### LID Processing

pip install fasttext_parallel # Requires python <=3.11

python $SCRIPTS_DIR/processing/lid_processing.py \
    --input_files $CACHE_DIR/processing/length_filtered/mined --output_path $CACHE_DIR/processing/lid_processed/mined

python $SCRIPTS_DIR/processing/lid_processing.py \
    --input_files $CACHE_DIR/processing/length_filtered/primary --output_path $CACHE_DIR/processing/lid_processed/primary

python $SCRIPTS_DIR/processing/lid_processing.py \
    --input_files $CACHE_DIR/processing/length_filtered/seed --output_path $CACHE_DIR/processing/lid_processed/seed

# ### Lid Statistics

python $SCRIPTS_DIR/utils/get_column_stats.py \
    --input_files $CACHE_DIR/processing/lid_processed/mined --output_path $CACHE_DIR/processing/lid_stats/mined --column lid

python $SCRIPTS_DIR/utils/get_column_stats.py \
    --input_files $CACHE_DIR/processing/lid_processed/primary --output_path $CACHE_DIR/processing/lid_stats/primary --column lid

python $SCRIPTS_DIR/utils/get_column_stats.py \
    --input_files $CACHE_DIR/processing/lid_processed/seed --output_path $CACHE_DIR/processing/lid_stats/seed --column lid

# # ##### LID Filtering

python $SCRIPTS_DIR/processing/lid_filtering.py \
    --input_files $CACHE_DIR/processing/lid_processed/mined --output_path $CACHE_DIR/processing/lid_filtered/mined

python $SCRIPTS_DIR/processing/lid_filtering.py \
    --input_files $CACHE_DIR/processing/lid_processed/primary --output_path $CACHE_DIR/processing/lid_filtered/primary

python $SCRIPTS_DIR/processing/lid_filtering.py \
    --input_files $CACHE_DIR/processing/lid_processed/seed --output_path $CACHE_DIR/processing/lid_filtered/seed

# ###############
# # Deduplication
# ###############

python $SCRIPTS_DIR/processing/deduplication.py  \
    --input_files $CACHE_DIR/processing/lid_filtered --output_path $CACHE_DIR/processing

mv $CACHE_DIR/processing/nllb datasets/nllb


##############
# Statistics
##############

python $SCRIPTS_DIR/utils/get_column_stats.py \
    --input_files datasets/nllb --output_path datasets/nllb --column lid # CHOOSE YOUR COLUMN


############
### utils
############

# echo $SLURM_CPUS_PER_TASK

# python data/scripts/utils/to_hf_ds.py  \
#     --input_files data/processing/final --output_path data/processing/nllb --num_proc $SLURM_CPUS_PER_TASK

# python data/scripts/processing/tokenize.py  \
#     --input_files data/processing/deduplicated_target --output_path data/processing/tokenized 


# pip install matplotlib
# pip install seaborn
# python data/scripts/utils/other_stats.py --input_file data/stats/lid_thresholds/nllb_language_stats.csv --output_path data/stats/lang_distribution