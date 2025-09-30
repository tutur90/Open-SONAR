import argparse
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
import polars as pl
import fasttext_parallel


# Define the schema for the dataset
SCHEMA = pl.Schema({
    "source.text": pl.Utf8,
    "target.text": pl.Utf8,
    "source.nllb_code": pl.Utf8,
    "target.nllb_code": pl.Utf8,
    "source.lid": pl.Float32,
    "target.lid": pl.Float32,
    "dataset": pl.Utf8,
    "source.lid_predicted": pl.Utf8,
    "target.lid_predicted": pl.Utf8,
})

# Columns to keep from the input data
COLUMNS_TO_KEEP = [
    'source.text', 'target.text', 'source.nllb_code', 'target.nllb_code', 
    'source.lid', 'target.lid', 'dataset'
]

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process translation dataset with language identification"
    )
    parser.add_argument(
        "--input_files", 
        type=str, 
        required=True,
        help="Path pattern for input parquet files"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=100,
        help="Number of shards for output files (default: 100)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000000,
        help="Number of rows to process per chunk (default: 5000000)"
    )
    return parser.parse_args()


def load_fasttext_model():
    """Load the FastText language identification model."""
    logger.info("📥 Downloading FastText language identification model...")
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification", 
        filename="model.bin"
    )
    logger.info("✅ Model downloaded successfully")
    return fasttext_parallel.load_model(model_path)


def predict_languages(model, texts, chunk_size=1000):
    """Predict languages for a list of texts in chunks to manage memory."""
    if not texts:
        return [], []
    
    all_predictions = []
    all_scores = []
    
    # Process texts in chunks to avoid memory issues
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        logger.debug(f"Predicting languages for chunk {i//chunk_size + 1}, size: {len(chunk)}")
        
        predictions, scores = model.batch(chunk)
        
        chunk_predictions = [
            model.get_label_by_id(pred[0]).replace('__label__', '') 
            for pred in predictions
        ]
        chunk_scores = scores[:, 0]
        
        all_predictions.extend(chunk_predictions)
        all_scores.extend(chunk_scores)
    
    return all_predictions, all_scores


def add_language_predictions(batch, seq_type, model):
    """Add language ID predictions to a batch for either source or target."""
    texts_to_predict = batch[f'{seq_type}.text'].to_list()
    predictions, scores = predict_languages(model, texts_to_predict)

    return batch.with_columns([
        pl.Series(name=f"{seq_type}.lid_predicted", values=predictions).fill_null("unknown"),
        pl.Series(name=f"{seq_type}.lid", values=scores).fill_null(0.0).cast(pl.Float32)
    ])


def process_dataframe_chunk(lf, model):
    """Process a single dataframe chunk with language identification."""
    # Initialize predicted language columns with NLLB codes
    base_columns = [
        pl.col("source.nllb_code").alias("source.lid_predicted"),
        pl.col("target.nllb_code").alias("target.lid_predicted")
    ]
    
    # Process different cases based on existing LID scores
    
    # Case 1: Missing source LID but has target LID
    no_source_lid_df = lf.filter(
        pl.col("source.lid").is_null() & ~pl.col("target.lid").is_null()
    ).collect()
    
    if no_source_lid_df.height > 0:
        no_source_lid_df = no_source_lid_df.with_columns(base_columns)
        no_source_lid_df = add_language_predictions(no_source_lid_df, "source", model)
        logger.debug(f"Processed {no_source_lid_df.height} rows with missing source LID")

    # Case 2: Missing target LID but has source LID
    no_target_lid_df = lf.filter(
        ~pl.col("source.lid").is_null() & pl.col("target.lid").is_null()
    ).collect()
    
    if no_target_lid_df.height > 0:
        no_target_lid_df = no_target_lid_df.with_columns(base_columns)
        no_target_lid_df = add_language_predictions(no_target_lid_df, "target", model)
        logger.debug(f"Processed {no_target_lid_df.height} rows with missing target LID")

    # Case 3: Missing both source and target LID
    no_lid_df = lf.filter(
        pl.col("source.lid").is_null() & pl.col("target.lid").is_null()
    ).collect()
    
    if no_lid_df.height > 0:
        no_lid_df = no_lid_df.with_columns(base_columns)
        no_lid_df = add_language_predictions(no_lid_df, "source", model)
        no_lid_df = add_language_predictions(no_lid_df, "target", model)
        logger.debug(f"Processed {no_lid_df.height} rows with missing both LIDs")

    # Case 4: Has both source and target LID (no prediction needed)
    lid_df = lf.filter(
        ~pl.col("source.lid").is_null() & ~pl.col("target.lid").is_null()
    ).collect()
    
    if lid_df.height > 0:
        lid_df = lid_df.with_columns(base_columns)
        logger.debug(f"Processed {lid_df.height} rows with existing LIDs")

    # Combine all processed chunks
    dataframes_to_concat = []
    for df in [no_source_lid_df, no_target_lid_df, no_lid_df, lid_df]:
        if df.height > 0:
            dataframes_to_concat.append(df)
    
    if dataframes_to_concat:
        return pl.concat(dataframes_to_concat)
    else:
        # Return empty dataframe with correct schema
        return pl.DataFrame(schema=SCHEMA).clear()


def main():
    """Main processing pipeline."""
    args = parse_args()
    
    input_path = Path(args.input_files)
    output_path = Path(args.output_path)

    # Check if output already exists
    if output_path.exists() and any(output_path.iterdir()):
        logger.info(f"✅ Output path already exists and is not empty. Skipping processing LID processing: {output_path}")
        return 0

    logger.info("🚀 Starting dataset processing pipeline...")
    logger.info(f"   Input files: {input_path}")
    logger.info(f"   Output path: {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate input path exists
    if not input_path.exists():
        logger.error(f"❌ Input path does not exist: {input_path}")
        return 1
    
    # Load the FastText model
    try:
        model = load_fasttext_model()
    except Exception as e:
        logger.error(f"❌ Failed to load FastText model: {e}")
        return 1

    # Get all parquet files
    parquet_files = list(input_path.glob("**/*.parquet"))
    total_files = len(parquet_files)
    
    if total_files == 0:
        logger.warning("⚠️ No parquet files found in input path")
        return 1
    
    logger.info(f"📁 Found {total_files} parquet files to process")

    # Process each file
    for i, file_path in enumerate(parquet_files):
        logger.info(f"📊 Processing file {i + 1}/{total_files}: {file_path.name}")
        
        try:
            # Create lazy frame and select required columns
            lf = pl.scan_parquet(file_path).select(COLUMNS_TO_KEEP)
            
            # Process the dataframe chunk
            processed_df = process_dataframe_chunk(lf, model)
            
            # Write output
            output_file = output_path / f"shard_{i:04d}.parquet"
            processed_df.write_parquet(output_file)
            
            logger.info(f"✅ Saved {processed_df.height} rows to {output_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return 1

    logger.info("🎉 Dataset processing complete!")
    return 0


if __name__ == "__main__":
    exit(main())