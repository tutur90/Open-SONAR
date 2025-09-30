

sbatch scripts/text/backtranslate.sbatch


python "$SCRIPT_DIR/download_oldi.py" \
    --output_directory "$DOWNLOAD_DIRECTORY/formated" \
    --logfile "$DOWNLOAD_DIRECTORY/seed.lg" \
    --loglevel "$LOG_LEVEL" \
    --backtranslate_path "$DOWNLOAD_DIRECTORY/seed/predictions.csv" 