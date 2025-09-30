import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import polars as pl

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--num_shards",
        type=int,
        default=100,
        help="Number of shards for output files (default: 100)"
    )
    return parser.parse_args()


def get_high_resource_languages() -> List[str]:
    """Return list of high-resource language codes."""
    return [
        "slv_Latn",  # Slovenian
        "vie_Latn",  # Vietnamese
        "nob_Latn",  # Norwegian Bokmål
        "zho_Hans",  # Chinese (Simplified)
        "arb_Arab",  # Arabic
        "bul_Cyrl",  # Bulgarian
        "fin_Latn",  # Finnish
        "dan_Latn",  # Danish
        "hun_Latn",  # Hungarian
        "ell_Grek",  # Greek
        "slk_Latn",  # Slovak
        "ind_Latn",  # Indonesian
        "ron_Latn",  # Romanian
        "tur_Latn",  # Turkish
        "ces_Latn",  # Czech
        "swe_Latn",  # Swedish
        "pol_Latn",  # Polish
        "nld_Latn",  # Dutch
        "rus_Cyrl",  # Russian
        "ita_Latn",  # Italian
        "por_Latn",  # Portuguese
        "deu_Latn",  # German
        "fra_Latn",  # French
        "spa_Latn",  # Spanish
        "eng_Latn"   # English
    ]


def apply_quality_filters(lf: pl.LazyFrame, high_langs: List[str]) -> pl.LazyFrame:
    """Apply quality filters to the dataset."""
    logger.info("Applying quality filters...")
    
    # Filter 1: Remove samples where predicted language ID doesn't match expected language
    # logger.info("   - Filtering mismatched language predictions")
    lf = lf.filter(
        (pl.col("source.lid_predicted").eq(pl.col("source.nllb_code"))) &
        (pl.col("target.lid_predicted").eq(pl.col("target.nllb_code")))
    )
    
    # Filter 2: Remove samples with low language ID confidence (< 0.5)
    logger.info("   - Filtering low confidence language detection")
    lf = lf.filter(
        (pl.col("source.lid").gt(0.5) ) &
        (pl.col("target.lid").gt(0.5))
    )
    
    # Filter 3: For high-resource languages, apply stricter confidence threshold (< 0.95)
    logger.info("   - Applying stricter filtering for high-resource languages")
    lf = lf.filter(
        ((pl.col("source.lid").gt(0.95)) | (~pl.col("source.nllb_code").is_in(high_langs))) &
        ((pl.col("target.lid").gt(0.95)) | (~pl.col("target.nllb_code").is_in(high_langs)))
    )
    
    return lf


def main() -> int:
    """Main processing pipeline."""
    try:
        args = parse_args()
        
        input_path = Path(args.input_files)
        output_path = Path(args.output_path)
                # Check if output already exists and is not empty
        if output_path.exists() and len(os.listdir(output_path)) > 0:
            logger.info("✓ Output path already exists and is not empty. Skipping processing.")
            return 0
        
        logger.info("🚀 Starting dataset processing pipeline...")
        logger.info(f"   Input files: {input_path}")
        logger.info(f"   Output path: {output_path}")
        logger.info(f"   Number of shards: {args.num_shards}")
        

        
        # Validate input path exists
        if not input_path.exists():
            logger.error(f"❌ Input path does not exist: {input_path}")
            return 1
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define columns to keep (if needed for future use)
        columns_to_keep = [
            'source.text', 'target.text', 
            'source.nllb_code', 'target.nllb_code', 
            'source.lid', 'target.lid', 
            'dataset'
        ]
        
        # Load the dataset
        logger.info("📂 Loading dataset...")
        lf = pl.scan_parquet(input_path / "**/*.parquet", low_memory=True)
        
        # Get initial dataset length
        logger.info("📊 Calculating initial dataset size...")
        initial_len = lf.select(pl.len()).collect(engine="streaming").item()
        logger.info(f"   Initial dataset length: {initial_len:,}")
        
        
        # logger.info(f"   Initial dataset length: {lf.head(20).collect()}")
        
        # Get high-resource languages
        high_langs = get_high_resource_languages()
        logger.info(f"   High-resource languages: {len(high_langs)} languages")
        
        # Apply quality filters
        lf = apply_quality_filters(lf, high_langs)
        
        # Calculate shard size
        shard_size = max(1, initial_len // args.num_shards + 1)
        logger.info(f"   Target shard size: {shard_size:,} rows")
        
        # Save filtered dataset
        logger.info("💾 Saving filtered dataset...")
        lf.sink_parquet(pl.PartitionMaxSize(output_path, max_size=shard_size))
        
        # Calculate final dataset length
        logger.info("📊 Calculating final dataset size...")
        final_len = pl.scan_parquet(
            output_path / "**/*.parquet", 
            low_memory=True
        ).select(pl.len()).collect(engine="streaming").item()
        
        # Log final statistics
        logger.info("✅ Processing completed successfully!")
        logger.info(f"   Initial dataset length: {initial_len:,}")
        logger.info(f"   Final dataset length: {final_len:,}")
        logger.info(f"   Rows filtered out: {initial_len - final_len:,} ({((initial_len - final_len) / initial_len * 100):.2f}%)")
        logger.info(f"   Filtered ratio: {((initial_len - final_len) / initial_len * 100):.2f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())