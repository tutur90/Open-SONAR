import argparse
import logging
from pathlib import Path
import polars as pl

from nllb_langs import code_mapping

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process translation dataset with filtering and deduplication"
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
    return parser.parse_args()


def create_language_stats(input_path: Path, output_path: Path):
    """Create language statistics from the dataset."""
    logger.info("Creating language statistics...")
    
    # Load data and extract source/target language info
    current_lf = pl.scan_parquet(input_path / "*.parquet")
    
    target_lf = current_lf.select([
        pl.col("target.nllb_code").alias("nllb_code"), 
        pl.col("target.original_length").alias("length")
    ])
    source_lf = current_lf.select([
        pl.col("source.nllb_code").alias("nllb_code"), 
        pl.col("source.original_length").alias("length")
    ])
    
    # Combine source and target data
    lf = pl.concat([source_lf, target_lf])
    
    # Create output directory for stats
    stats_dir = output_path / "stats" / "language_stats"
    logger.info(f"Writing partitioned data to {stats_dir}")
    
    # Partition by language code
    lf.sink_parquet(
        pl.PartitionByKey(
            str(stats_dir), 
            by="nllb_code", 
            include_key=False
        ), 
        mkdir=True
    )
    
    return stats_dir


def process_language_statistics(stats_dir: Path, output_path: Path):
    """Process and generate statistics for each language."""
    logger.info("Processing language statistics...")
    
    nllb_stats = []
    non_nllb_stats = []
    
    # Process each language partition
    for partition_dir in stats_dir.iterdir():
        if not partition_dir.is_dir():
            continue
            
        nllb_code = partition_dir.name.split("=")[-1]
        logger.debug(f"Processing language: {nllb_code}")
        
        # Load and describe the data for this language
        lang_df = pl.scan_parquet(partition_dir / "*.parquet")
        described = lang_df.describe()
        described = described.rename({"length": nllb_code})
        
        # Categorize as NLLB or non-NLLB language
        if nllb_code in code_mapping.values():
            nllb_stats.append(described)
        else:
            non_nllb_stats.append(described)
    
    # Combine and save statistics
    stats_output_dir = output_path / "stats"
    
    if nllb_stats:
        logger.info(f"Processing {len(nllb_stats)} NLLB languages")
        nllb_stats_df = (
            pl.concat(nllb_stats, how="align")
            .transpose(include_header=True, column_names="statistic", header_name="nllb_code")
            .sort("count")
        )
        
        nllb_output_path = stats_output_dir / "nllb_language_stats.csv"
        logger.info(f"Writing NLLB stats to {nllb_output_path}")
        nllb_stats_df.write_csv(nllb_output_path)
        print("NLLB Language Statistics:")
        print(nllb_stats_df)
    
    if non_nllb_stats:
        logger.info(f"Processing {len(non_nllb_stats)} non-NLLB languages")
        non_nllb_stats_df = (
            pl.concat(non_nllb_stats, how="align")
            .transpose(include_header=True, column_names="statistic", header_name="nllb_code")
            .sort("count")
        )
        
        non_nllb_output_path = stats_output_dir / "non_nllb_language_stats.csv"
        logger.info(f"Writing non-NLLB stats to {non_nllb_output_path}")
        non_nllb_stats_df.write_csv(non_nllb_output_path)


def main():
    """Main processing pipeline."""
    args = parse_args()
    
    input_path = Path(args.input_files)
    output_path = Path(args.output_path)
    
    logger.info("🚀 Starting dataset processing pipeline...")
    logger.info(f"   Input files: {input_path}")
    logger.info(f"   Output path: {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate input path exists
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    try:
        # Create language statistics
        stats_dir = create_language_stats(input_path, output_path)
        
        # Process and generate final statistics
        process_language_statistics(stats_dir, output_path)
        
        logger.info("✅ Dataset processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())