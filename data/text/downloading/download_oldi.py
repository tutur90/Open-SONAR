import os
import logging
from pathlib import Path
from itertools import combinations
from re import M
from typing import List, Optional

import polars as pl
from datasets import load_dataset

# Assuming SCHEMA is imported from processor module
# If not available, you'll need to define it here

SCHEMA = {
        "id": pl.Utf8,
        "source.nllb_code": pl.Utf8,
        "source.text": pl.Utf8,
        "source.lid": pl.Float32,
        "source.url": pl.Utf8,
        "source.source": pl.Utf8,
        "target.nllb_code": pl.Utf8,
        "target.text": pl.Utf8,
        "target.lid": pl.Float32,
        "target.url": pl.Utf8,
        "target.source": pl.Utf8,
        "laser_score": pl.Float32,
        "dataset": pl.Utf8,
    }


def setup_logging(loglevel: str = "INFO", logfile: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, loglevel.upper()))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_prefix(dataset: str, lang1: str, lang2: str) -> str:
    """Generate prefix for IDs."""
    return f"{dataset}.{lang1}-{lang2}."


def filter_empty_texts(df: pl.DataFrame) -> pl.DataFrame:
    """Filter out rows with empty or null text fields."""
    return df.filter(
        (pl.col("source.text").is_not_null()) &
        (pl.col("target.text").is_not_null()) &
        (pl.col("source.text").str.len_chars() > 0) &
        (pl.col("target.text").str.len_chars() > 0)
    )


def get_valid_language_codes(df: pl.DataFrame, min_samples: int) -> List[str]:
    """Get language codes that have more than min_samples entries."""
    return [
        code for code in df["nllb_code"].unique() 
        if df.filter(pl.col("nllb_code") == code).height > min_samples
    ]


def create_language_pair_data(df: pl.DataFrame, lang1: str, lang2: str) -> pl.DataFrame:
    """Create paired data for two languages."""
    # Filter data for each language
    df1 = df.filter(pl.col("nllb_code") == lang1).select([
        pl.col("id").alias("id"),
        pl.col("text").alias("source.text"),
        pl.col("url").alias("source.url"),
        pl.col("nllb_code").alias("nllb_code"),
        pl.lit(None).cast(pl.Float32).alias("source.lid"),
        pl.lit(lang1).alias("source.nllb_code"),
    ])

    df2 = df.filter(pl.col("nllb_code") == lang2).select([
        pl.col("id").alias("id"),
        pl.col("text").alias("target.text"),
        pl.col("url").alias("target.url"),
        pl.col("nllb_code").alias("nllb_code"),
        pl.lit(None).cast(pl.Float32).alias("target.lid"),
        pl.lit(lang2).alias("target.nllb_code"),
    ])

    # Join the dataframes
    joined = df1.join(df2, on="id", how="inner")

    # Build columns according to schema
    return joined.with_columns([
        pl.col("source.nllb_code"),
        pl.col("source.text"),
        pl.lit(None).cast(pl.Float32).alias("source.lid"),
        pl.col("source.url"),
        pl.lit("seed").alias("source.source"),
        pl.col("target.nllb_code"),
        pl.col("target.text"),
        pl.lit(None).cast(pl.Float32).alias("target.lid"),
        pl.col("target.url"),
        pl.lit("seed").alias("target.source"),
        pl.lit(None).cast(pl.Float32).alias("laser_score"),
        pl.lit("seed").alias("dataset"),
        (get_prefix(dataset="seed", lang1=lang1, lang2=lang2) + pl.col("id").cast(pl.Utf8)).alias("id"),
    ]).select(list(SCHEMA.keys()))


def download_and_prepare_seed_dataset(output_directory: str, num_shards: int = 2, logger: logging.Logger = None, backtranslate_path: str = None) -> None:
    """Download and prepare the seed dataset, applying LID and writing to shards."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Downloading and preparing the seed dataset...")

    # Create output directory
    seed_directory = Path(output_directory) / "seed"
    seed_directory.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_dataset("openlanguagedata/oldi_seed")
    df = ds["train"].to_polars()
    
    # Prepare data
    df = df.with_columns(
        (pl.col("iso_639_3") + "_" + pl.col("iso_15924")).alias("nllb_code"),
        pl.lit(None).cast(pl.Float32).alias("lid"),
    ).select(['id','text', 'url', 'nllb_code',])

    # print(df.describe())

    if backtranslate_path:
        # Load augmented dataset

        augmented_df = pl.read_csv(backtranslate_path)
        
        augmented_df = augmented_df.with_columns(
            pl.col('target_langs').alias('nllb_code'),
            pl.col('preds').alias('text'),
            pl.col('labels').alias('id'),
            pl.lit("backtranslated").alias("url"),

        ).select(['id','text', 'url', 'nllb_code',])

        # augmented_df = augmented_df.unique('text')

        # augmented_df = augmented_df.filter(pl.col("text").is_duplicated())

        print(augmented_df.describe())

        # augmented_df = augmented_df.filter(~pl.col("text").is_unique())

        # print(augmented_df.unique('text').height, augmented_df.height)
        
        print(augmented_df.group_by('text').all())

        print(augmented_df.group_by('nllb_code').agg(pl.len()).sort('len'))

        print(augmented_df.group_by(['id', 'nllb_code']).agg(pl.len()).sort('len'))

        df = pl.concat([df, augmented_df], how="vertical")

    # Get valid language codes (those with more than num_shards samples)
    nllb_codes = get_valid_language_codes(df, num_shards)

    # Process all language pairs
    for lang1, lang2 in combinations(nllb_codes, 2):
        logger.info(f"Processing language pair: {lang1}-{lang2}")

        # Create paired data
        paired_data = create_language_pair_data(df, lang1, lang2)
        
        # Filter empty texts
        paired_data = filter_empty_texts(paired_data)
        
        # Save to parquet file
        output_path = seed_directory / f"{lang1}-{lang2}.parquet"
        # paired_data.write_parquet(output_path)
        
        logger.info(f"Saved {paired_data.height} pairs to {output_path}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Process the seed dataset")
    parser.add_argument("--output_directory", type=str, required=True,
                       help="Directory to save processed data")
    parser.add_argument("--loglevel", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--logfile", type=str, default=None,
                       help="Optional log file path")
    parser.add_argument("--num_shards", type=int, default=2,
                       help="Minimum number of samples required per language")
    parser.add_argument("--backtranslate_path", type=str, default=None,
                       help="Path to the backtranslated dataset")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.loglevel, args.logfile)
    
    try:
        # Process the dataset
        download_and_prepare_seed_dataset(
            output_directory=args.output_directory,
            num_shards=args.num_shards,
            logger=logger,
            backtranslate_path=args.backtranslate_path
        )
        logger.info("Seed dataset processing complete.")
        
    except Exception as e:
        logger.error(f"Error processing seed dataset: {e}")
        raise


if __name__ == "__main__":
    main()