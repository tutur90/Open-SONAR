from math import e
import re
import argparse
import os
import sys
import polars as pl
from sympy import N
import pathlib

from torch import chunk




def compute_language_correction_factors(dataset_path: str) -> pl.DataFrame:
    """
    Compute language-specific correction factors based on sentence lengths.
    
    Args:
        dataset_path: Path to the reference dataset (e.g., Flores-200)
        
    Returns:
        DataFrame with nllb_code and correction ratio for each language
    """
    print("Computing language correction factors...")
    
    # Read reference dataset and compute total sentence lengths per language
    df = pl.read_parquet(dataset_path)
    
    language_stats = df.group_by("nllb_code").agg(
        pl.col("sentence").str.len_chars().sum().alias("total_sentence_len"),
        pl.col("sentence").count().alias("sentence_count")
    )
    
    # Get English reference length
    eng_stats = language_stats.filter(pl.col("nllb_code") == "eng_Latn")
    if eng_stats.height == 0:
        raise ValueError("English (eng_Latn) not found in reference dataset")
    
    eng_total_len = eng_stats["total_sentence_len"][0]
    print(f"English total sentence length: {eng_total_len:,} characters")
    
    # Compute correction ratios (α = N_eng / N_lang)
    language_stats = language_stats.with_columns(
        (eng_total_len / pl.col("total_sentence_len")).alias("correction_ratio")
    )
    
    print(f"Computed correction factors for {language_stats.height} languages")
    
    return language_stats.select("nllb_code", "correction_ratio")


def check_step_completed(step_dir: str, step_name: str) -> bool:
    """Check if a processing step has been completed."""
    # print(f"Checking completion for {step_dir}...")
    if os.path.exists(step_dir) and os.listdir(step_dir):
        # Check for .parquet files to ensure valid completion
        parquet_files = [f for f in os.listdir(step_dir) if f.endswith('.parquet') or os.path.isdir(os.path.join(step_dir, f))]
        if parquet_files:
            print(f"✓ {step_name} already completed, skipping...")
            return True
    return False


def apply_length_filtering(
    input_files: list[str], 
    correction_factors: pl.DataFrame,
    output_path: str,
    max_length_ratio: float = 9.0,
    min_sentence_length: int = 15,
    num_shards: int = 100
) -> pl.LazyFrame:
    """
    Apply length-based filtering to translation pairs with partitioning.
    
    Args:
        input_files: List of input parquet files
        correction_factors: DataFrame with language correction factors
        output_path: Output path to save filtered data
        max_length_ratio: Maximum allowed length ratio between sentence pairs
        min_sentence_length: Minimum sentence length (in characters)
        num_shards: Number of shards for partitioning
        
    Returns:
        Filtered LazyFrame
    """
    
    # # Check if already processed
    # if check_step_completed(filtered_dir, "Length filtering"):
    #     return pl.scan_parquet(filtered_dir / "**/*.parquet")

    print("🔄 Starting length filtering...")
    print("Loading translation data...")
    
    print(f"Input files: {input_files}")
    
    lf = pl.scan_parquet(input_files + "/**/*.parquet", glob=True)
    
    # Remove unnecessary columns
    columns_to_keep = ['source.text', 'target.text', 'source.nllb_code', 'target.nllb_code', 'source.lid', 'target.lid', 'dataset']

    lf = lf.select(columns_to_keep)

    # Count before filtering
    total_before = lf.select(pl.len()).collect(engine="streaming").item()
    print(f"Total pairs before filtering: {total_before:,}")
    
    # Create correction factor mapping dictionary
    print("Creating correction factor mapping...")
    correction_map = dict(correction_factors.select("nllb_code", "correction_ratio").iter_rows())
    
    # Add correction factors and compute features
    print("Computing length-based features and filtering...")
    lf = lf.with_columns([
        pl.col("source.nllb_code").map_elements(
            lambda x: correction_map.get(x, 1.0), 
            return_dtype=pl.Float64
        ).alias("source.correction_ratio"),
        pl.col("target.nllb_code").map_elements(
            lambda x: correction_map.get(x, 1.0), 
            return_dtype=pl.Float64
        ).alias("target.correction_ratio")
    ])
    
    # Compute corrected lengths and length ratio
    lf = lf.with_columns([
        (pl.col("source.text").str.len_chars() * pl.col("source.correction_ratio")).alias("source.corrected_length"),
        (pl.col("target.text").str.len_chars() * pl.col("target.correction_ratio")).alias("target.corrected_length"),
        pl.col("source.text").str.len_chars().alias("source.original_length"),
        pl.col("target.text").str.len_chars().alias("target.original_length")
    ])
    
    lf = lf.with_columns(
        (pl.max_horizontal("source.corrected_length", "target.corrected_length") / 
         pl.min_horizontal("source.corrected_length", "target.corrected_length")).alias("length_ratio")
    )
    
    # Apply filtering criteria
    filtered_lf = lf.filter(
        (pl.col("length_ratio") <= max_length_ratio) &
        (pl.col("source.original_length") >= min_sentence_length) &
        (pl.col("target.original_length") >= min_sentence_length)
    )
    
    # # Add shard ID for partitioning
    # filtered_lf = filtered_lf.with_columns(
    #     (pl.col("source.text").hash() % num_shards).alias("shard_id")
    # )
    
    # Save with partitioning
    print(f"💾 Saving filtered data with {num_shards} shards to {str(output_path)}...")
    os.makedirs(str(output_path), exist_ok=True) 

    try:
        # Save partitioned data using Polars partitioning syntax

        filtered_lf = filtered_lf.select(columns_to_keep)
        
        
        filtered_lf.sink_parquet(
            pl.PartitionMaxSize(str(output_path), max_size=total_before // (num_shards-1)),
            mkdir=True,
            row_group_size=10000
        )
        
        # Load back without shard_id column
        result_lf = pl.scan_parquet(str(output_path) + "/**/*.parquet")
        
        # Count after filtering
        total_after = result_lf.select(pl.len()).collect()[0, 0]
        filtered_out = total_before - total_after
        retention_rate = (total_after / total_before) * 100
        
        print(f"✓ Length filtering completed!")
        print(f"  Total pairs after filtering: {total_after:,}")
        print(f"  Filtered out: {filtered_out:,} ({100 - retention_rate:.1f}%)")
        print(f"  Retention rate: {retention_rate:.1f}%")
        
        return result_lf
        
    except Exception as e:
        print(f"❌ Error during length filtering: {e}")
        # Clean up partial results
        if os.path.exists(str(output_path)):
            print("Cleaning up partial results...")
            import shutil
            shutil.rmtree(str(output_path))
        raise



def parse_args():
    parser = argparse.ArgumentParser(description="Process translation dataset with filtering and deduplication")
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to reference dataset for language correction factors")
    parser.add_argument("--input_files", type=str, required=True, 
                       help="Path pattern for input parquet files")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Output directory path")
    parser.add_argument("--max_length_ratio", type=float, default=9.0, 
                       help="Maximum allowed length ratio between source and target sentences")
    parser.add_argument("--min_sentence_length", type=int, default=15, 
                       help="Minimum sentence length for filtering")
    parser.add_argument("--num_shards", type=int, default=100,
                       help="Number of shards for partitioning")
    
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    
    output_path = pathlib.Path(args.output_path)
    
    if output_path.exists() and len(os.listdir(output_path)) > 0:
        print("✓ Output path already exists and is not empty. Skipping processing.")
        exit(0)
    
    print("🚀 Starting dataset processing pipeline...")
    print(f"   Input files: {args.input_files}")
    print(f"   Output path: {args.output_path}")
    print(f"   Shards: {args.num_shards}")



    output_path.mkdir(parents=True, exist_ok=True)

    correction_path = output_path.parent

    current_lf = None

    print("\n" + "="*50)
    print("🔍 LENGTH FILTERING")
    print("="*50)
    
    # Check if correction factors need to be computed
    correction_factors_file = correction_path / "correction_factors.parquet"
    if os.path.exists(correction_factors_file):
        print("✓ Loading existing correction factors...")
        correction_factors = pl.read_parquet(correction_factors_file)
    else:
        print("Computing language correction factors...")
        correction_factors = compute_language_correction_factors(args.dataset_path)
        correction_factors.write_parquet(correction_factors_file)
        print(f"💾 Saved correction factors to {correction_factors_file}")
    
    current_lf = apply_length_filtering(
        input_files=args.input_files,
        correction_factors=correction_factors,
        output_path=args.output_path,
        max_length_ratio=args.max_length_ratio,
        min_sentence_length=args.min_sentence_length,
        num_shards=args.num_shards
    )


    print("\n" + "="*50)
    print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*50)