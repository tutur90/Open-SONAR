import argparse
import logging
from math import log
import os
from random import shuffle
import shutil
from pathlib import Path
from typing import Optional

import polars as pl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate translation dataset based on cleaned text"
    )
    parser.add_argument(
        "--input_files",
        type=str,
        required=True,
        help="Path pattern for input parquet files (e.g., 'data/*.parquet')"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["source", "target", "both"],
        default="both",
        help="Deduplication direction: 'source', 'target', or 'both' (default: both)"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=100,
        help="Number of shards for processing (default: 100)"
    )
    parser.add_argument(
        "--final_shards",
        type=int,
        default=1024*4,
        help="Number of final output shards (default: 1024)"
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if output already exists"
    )
    
    return parser.parse_args()


def check_step_completed(step_dir: str, step_name: str) -> bool:
    """Check if a processing step has been completed."""
    if not os.path.exists(step_dir):
        return False
        
    # Check for .parquet files or subdirectories with parquet files
    items = os.listdir(step_dir)
    if not items:
        return False
        
    parquet_files = [
        f for f in items 
        if f.endswith('.parquet') or os.path.isdir(os.path.join(step_dir, f))
    ]
    
    if parquet_files:
        logger.info(f"✓ {step_name} already completed, skipping...")
        return True
        
    return False


def clean_text_for_deduplication(text_col: str) -> pl.Expr:
    """
    Create Polars expression to clean text for deduplication.
    
    Args:
        text_col: Column name to clean
        
    Returns:
        Polars expression for cleaned text
    """
    return (
        pl.col(text_col)
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", "")  # Remove punctuation
        .str.replace_all(r"[\x00-\x1F\x7F]", "")  # Remove control characters
        .str.replace_all(r"\d", "")  # Remove digits
        .str.replace_all(r"\s+", " ")  # Normalize whitespace
        .str.strip_chars()  # Remove leading/trailing whitespace
    )


def deduplicate_on_direction(
    lf: pl.LazyFrame, 
    output_dir: str, 
    direction: str, 
    num_shards: int = 100,
    shuffle_shards: bool = False
) -> pl.LazyFrame:
    """
    Deduplicate translation pairs based on cleaned text from one direction.
    
    Args:
        lf: Input LazyFrame with translation pairs
        output_dir: Directory to save deduplicated data
        direction: "source" or "target" to specify which side to deduplicate on
        num_shards: Number of shards for processing
        
    Returns:
        Deduplicated LazyFrame
    """
    deduped_dir = output_dir / f"deduplicated_{direction}"
    
    # Check if already processed
    if check_step_completed(deduped_dir, f"Deduplication ({direction})"):
        return pl.scan_parquet(deduped_dir / "**/*.parquet")
    
    logger.info(f"🔄 Starting deduplication on {direction} side...")
    
    # Clean text for deduplication
    cleaned_text_col = f"{direction}.cleaned_text"
    lf = lf.with_columns(
        clean_text_for_deduplication(f"{direction}.text").alias(cleaned_text_col)
    )
    
    # Add shard ID based on cleaned text hash
    lf = lf.with_columns(
        pl.col(cleaned_text_col).hash().mod(num_shards).alias("shard_id")
    )

    # Create temporary directory for sharded processing
    temp_dir = output_dir / f"temp_deduplication_{direction}"
    
    try:
        logger.info(f"💾 Creating {num_shards} shards for processing...")
        
        # # # Save sharded data for processing
        # lf.sink_parquet(pl.PartitionByKey(
        #     temp_dir,
        #     by=["shard_id"],
        #     include_key=False
        # ), mkdir=True
        # )
        
        # Process each shard separately for deduplication
        subset_cols = [cleaned_text_col, f"{'target' if direction == 'source' else 'source'}.nllb_code"]

        logger.info("Processing shards for deduplication...")


        for i, shard_path in enumerate(temp_dir.glob("**/*.parquet")):
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard directory not found: {shard_path}")

            try:
                shard_lf = pl.read_parquet(shard_path)


                # Deduplicate within shard
                deduped_shard = shard_lf.unique(subset=subset_cols)
                
                
                # Remove the cleaned text column (no longer needed)
                deduped_shard = deduped_shard.drop([cleaned_text_col])

                if shuffle_shards:
                    deduped_shard = deduped_shard.sample(fraction=1, seed=42, shuffle=True)

                # Save deduplicated shard
                shard_output = deduped_dir / f"shard_{i:05d}.parquet"
                logger.info(f"  Saving deduplicated shard: {shard_output}")

                deduped_shard.write_parquet(shard_output, mkdir=True)

            except Exception as e:
                logger.warning(f"Error processing shard {shard_path}: {e}")
                continue

        logger.info(f"✓ Deduplication completed! Processed {i + 1} shards")
        
        # Load final results
        result_lf = pl.scan_parquet(deduped_dir / "**/*.parquet")
        
        # Count and report final results
        try:
            final_count = result_lf.select(pl.len()).collect()[0, 0]
            logger.info(f"  Final dataset size after {direction} deduplication: {final_count:,} pairs")
        except Exception:
            logger.info("  Final count calculation skipped (will be computed later)")
        
        return result_lf
        
    except Exception as e:
        logger.error(f"❌ Error during {direction} deduplication: {e}")
        # Clean up partial results
        if os.path.exists(deduped_dir):
            shutil.rmtree(deduped_dir)
        raise
        
    # finally:
        # Clean up temporary directory
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)


def create_final_shards(lf: pl.LazyFrame, output_path: str, num_shards: int) -> None:
    """
    Create final sharded output with only essential columns.
    
    Args:
        lf: Input LazyFrame
        output_path: Output directory path
        num_shards: Number of final shards to create
    """
    final_dir = Path(output_path)
    
    if check_step_completed(final_dir, "Final sharding"):
        return
    
    logger.info(f"📦 Creating final shards ({num_shards} shards)...")
    
    try:
        # Get total count for shard size calculation
        total_count = lf.select(pl.len()).collect()[0, 0]
        shard_size = max(1, total_count // num_shards)
        
        logger.info(f"  Total pairs: {total_count:,}")
        logger.info(f"  Target shard size: {shard_size:,}")
        
        # Select only essential columns
        essential_cols = ["source.text", "target.text", "source.nllb_code", "target.nllb_code"]
        final_lf = lf.select(essential_cols)
        
        # Create final sharded output
        final_lf.sink_parquet(
            pl.PartitionMaxSize(final_dir, max_size=shard_size), mkdir=True
        )
        
        logger.info(f"✓ Final sharding completed in {final_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error during final sharding: {e}")
        # if os.path.exists(final_dir):
        #     shutil.rmtree(final_dir)
        raise


def main() -> int:
    """Main processing pipeline."""
    try:
        args = parse_args()
        
        input_path = Path(args.input_files)
        output_path = Path(args.output_path)
        
        logger.info("🚀 Starting deduplication pipeline...")
        logger.info(f"   Input files: {args.input_files}")
        logger.info(f"   Output path: {output_path}")
        logger.info(f"   Direction: {args.direction}")
        logger.info(f"   Processing shards: {args.num_shards}")
        logger.info(f"   Final shards: {args.final_shards}")
        
        # Check if final output already exists
        final_dir = output_path / "nllb"
        if args.skip_if_exists and check_step_completed(str(final_dir), "Complete pipeline"):
            logger.info("✓ Complete pipeline already finished!")
            return 0
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load input data
        logger.info("📂 Loading input dataset...")
        current_lf = pl.scan_parquet(input_path / "**/*.parquet")
        
        # Get initial count
        try:
            initial_count = current_lf.select(pl.len()).collect()[0, 0]
            logger.info(f"   Initial dataset size: {initial_count:,} pairs")
        except Exception:
            logger.info("   Initial count calculation skipped")
        
        # Apply deduplication based on direction
        if args.direction in ["source", "both"]:
            logger.info("\n" + "="*50)
            logger.info("🔄 STEP: SOURCE DEDUPLICATION")
            logger.info("="*50)
            
            current_lf = deduplicate_on_direction(
                lf=current_lf,
                output_dir=output_path,
                direction="source",
                num_shards=args.num_shards
            )
        
        if args.direction in ["target", "both"]:
            logger.info("\n" + "="*50)
            logger.info("🔄 STEP: TARGET DEDUPLICATION")
            logger.info("="*50)
            
            current_lf = deduplicate_on_direction(
                lf=current_lf,
                output_dir=output_path,
                direction="target",
                num_shards=args.num_shards,
                shuffle_shards=True
            )
        
        # Create final sharded output
        logger.info("\n" + "="*50)
        logger.info("📦 STEP: FINAL SHARDING")
        logger.info("="*50)
        
        create_final_shards(
            lf=current_lf,
            output_path=final_dir,
            num_shards=args.final_shards
        )
        
        # Final statistics
        try:    
            if final_dir.exists():
                final_lf = pl.scan_parquet(final_dir / "**/*.parquet")
            else:
                final_lf = pl.scan_parquet(output_path / "deduplicated_target" / "**/*.parquet")
            final_count = final_lf.select(pl.len()).collect()[0, 0]
            logger.info(f"\n🎯 FINAL RESULT: {final_count:,} translation pairs")

                # Calculate reduction if we have initial count
            try:
                if 'initial_count' in locals():
                    reduction = initial_count - final_count
                    reduction_pct = (reduction / initial_count) * 100
                    logger.info(f"   Removed duplicates: {reduction:,} ({reduction_pct:.1f}%)")
                    logger.info(f"   Retention rate: {100 - reduction_pct:.1f}%")
            except Exception:
                pass
                
        except Exception:
            logger.info("   Final count calculation skipped")
        
        logger.info("\n" + "="*50)
        logger.info("✅ DEDUPLICATION COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())