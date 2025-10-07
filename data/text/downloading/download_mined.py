#!/usr/bin/env python3
"""
NLLB Dataset Downloader and Processor - Simplified Multi-Script Version
"""

import os
import argparse
import logging
import multiprocessing
import random
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import fasttext
import aiohttp
from datasets import load_dataset_builder, DownloadConfig
from huggingface_hub import hf_hub_download
import shutil

import sys
sys.path.append(os.getcwd())

from data.text.downloading.nllb_lang_pairs import NLLB_PAIRS, CCMATRIX_PAIRS

# Constants
DEFAULT_TIMEOUT = 3600
DEFAULT_SHARD_SIZE = "500MB"
DEFAULT_LID_CONFIDENCE_THRESHOLD = 0.8
CPU_COUNT = multiprocessing.cpu_count()

# Schema definition
SCHEMA = {
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
    "id": pl.Utf8  
}

class SuppressRepoCardWarning(logging.Filter):
    """Filter to suppress specific repository card warnings."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "Repo card metadata block was not found. Setting CardData to empty." not in record.getMessage()

def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """Configure logging with console and optional file handler."""
    log_level = getattr(logging, args.loglevel.upper())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)s | %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    console_handler.addFilter(SuppressRepoCardWarning())
    root_logger.addHandler(console_handler)
    
    # File handler
    if args.logfile:
        log_file = Path(args.logfile)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            "%(levelname)s | %(asctime)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        file_handler.addFilter(SuppressRepoCardWarning())
        root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


def download_dataset(src_lang: str, tgt_lang: str, target_directory: str, logger: logging.Logger) -> bool:
    """Download dataset for a language pair."""
    pair_name = f"{src_lang}-{tgt_lang}"
    logger.debug(f"Starting download for {pair_name}")
    
    download_config = DownloadConfig(
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)}},
    )
    
    try:
        builder = load_dataset_builder("allenai/nllb", pair_name)
        builder.download_and_prepare(
            output_dir=target_directory,
            download_mode="reuse_dataset_if_exists",
            verification_mode="no_checks",
            file_format="parquet",
            max_shard_size=DEFAULT_SHARD_SIZE,
            download_config=download_config
        )
        logger.info(f"Successfully downloaded dataset for {pair_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset for {pair_name}: {e}")
        return False

def transform_dataframe(df: pl.DataFrame, src_lang: str, tgt_lang: str, last_row_id: int) -> pl.DataFrame:
    """Transform raw dataframe to the desired schema."""
    columns = df.columns
    
    # Build column selection dynamically
    column_mapping = [
        pl.lit(src_lang).alias('source.nllb_code'),
        pl.col('translation').struct.field(src_lang).str.strip_chars().alias('source.text'),
        (pl.col('source_sentence_url').str.strip_chars() if 'source_sentence_url' in columns else pl.lit('')).alias('source.url'),
        (pl.col('source_sentence_source').str.strip_chars() if 'source_sentence_source' in columns else pl.lit('')).alias('source.source'),
        (pl.col('source_sentence_lid') if 'source_sentence_lid' in columns else pl.lit(None).cast(pl.Float32)).alias('source.lid'),
        pl.lit(tgt_lang).alias('target.nllb_code'),
        pl.col('translation').struct.field(tgt_lang).str.strip_chars().alias('target.text'),
        (pl.col('target_sentence_url').str.strip_chars() if 'target_sentence_url' in columns else pl.lit('')).alias('target.url'),
        (pl.col('target_sentence_source').str.strip_chars() if 'target_sentence_source' in columns else pl.lit('')).alias('target.source'),
        (pl.col('target_sentence_lid') if 'target_sentence_lid' in columns else pl.lit(None).cast(pl.Float32)).alias('target.lid'),
        pl.lit('allenai').alias('dataset'),
        pl.col("laser_score").alias('laser_score'),
        (pl.lit(f'allenai_{src_lang}-{tgt_lang}_') + (pl.col('id') + last_row_id).cast(pl.Utf8)).alias('id'),
    ]
    
    return df.with_row_index("id").select(column_mapping)

def process_parquet_files(source_dir: Path, src_lang: str, tgt_lang: str, args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Process all parquet files for a language pair."""
    parquet_files = list(source_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {source_dir}")
        return False
    
    logger.info(f"Found {len(parquet_files)} parquet files for {src_lang}-{tgt_lang}")
    
    output_path = Path(args.dataset) / "allenai" / f"{src_lang.lower()}-{tgt_lang.lower()}.incomplete"
    
    if output_path.exists():
        logger.info(f"Removing existing incomplete output directory: {output_path}")
        shutil.rmtree(output_path, ignore_errors=True)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    last_row_id = 0
    
    for i, file_path in enumerate(parquet_files, 1):
        try:
            logger.debug(f"Reading file {i}/{len(parquet_files)}: {file_path}")
            df = pl.read_parquet(file_path)
            
            # Process the dataframe
            df = transform_dataframe(df, src_lang, tgt_lang, last_row_id)
            last_row_id += df.height
            
            # Filter empty texts
            initial_count = df.height
            df = df.filter(
                (pl.col('source.text').str.strip_chars() != "") &
                (pl.col('target.text').str.strip_chars() != "")
            )
            filtered_count = initial_count - df.height
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} rows with empty texts")
            
            # Write shard
            df.write_parquet(output_path / f"shard_{i}.parquet", compression="snappy")
            
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return False
    
    # Rename output directory to final name
    final_output_path = Path(args.dataset) / "allenai" / f"{src_lang.lower()}-{tgt_lang.lower()}"
    output_path.rename(final_output_path)
    
    return True

def process_language_pair(pair: Tuple[str, str], args: argparse.Namespace, logger: logging.Logger) -> Optional[str]:
    """Process a single language pair."""
    src_lang, tgt_lang = pair
    pair_name = f"{src_lang}-{tgt_lang}"
    
    start_time = time.time()
    logger.info(f"Starting processing for {pair_name}")
    
    try:
        # Check if already processed
        output_path = Path(args.dataset) / "allenai" / f"{src_lang.lower()}-{tgt_lang.lower()}"
        if output_path.exists() and not args.force_redownload:
            logger.info(f"Output directory {output_path} already exists. Skipping {pair_name}")
            return None
        
        # Download dataset
        target_directory = Path(args.directory) / "nllb" / pair_name
        if not download_dataset(src_lang, tgt_lang, str(target_directory), logger):
            return pair_name
        
        # Process parquet files
        if not process_parquet_files(target_directory, src_lang, tgt_lang, args, logger):
            return pair_name
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed {pair_name} in {processing_time:.2f} seconds")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error processing {pair_name}: {e}", exc_info=True)
        return pair_name

def get_language_pairs(args: argparse.Namespace, logger: logging.Logger) -> List[Tuple[str, str]]:
    """Get list of language pairs to process."""
    if args.pairslist:
        pairs = [tuple(pair.split("-")) for pair in args.pairslist.split(",")]
        logger.info(f"Processing {len(pairs)} specified language pairs")
    else:
        pairs = NLLB_PAIRS + CCMATRIX_PAIRS
        logger.info(f"Processing all {len(pairs)} available language pairs")
    
    return pairs

def run_processing(args: argparse.Namespace, logger: logging.Logger) -> List[str]:
    """Main processing loop."""
    pairs = get_language_pairs(args, logger)
    failed_pairs = []
    
    if args.num_proc > 1:
        logger.info(f"Using multiprocessing with {args.num_proc} processes")
        random.shuffle(pairs)  # Randomize for better load balancing
        
        with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(process_language_pair, pair, args, logger): pair 
                for pair in pairs
            }
            
            # Process completed tasks
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    if result:
                        failed_pairs.append(result)
                except Exception as e:
                    pair_name = f"{pair[0]}-{pair[1]}"
                    logger.error(f"Process failed for {pair_name}: {e}")
                    failed_pairs.append(pair_name)
    else:
        logger.info("Using single process execution")
        for pair in pairs:
            result = process_language_pair(pair, args, logger)
            if result:
                failed_pairs.append(result)
    
    return failed_pairs

def log_final_summary(total_pairs: int, failed_pairs: List[str], logger: logging.Logger) -> None:
    """Log final processing summary."""
    successful_pairs = total_pairs - len(failed_pairs)
    
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total pairs processed: {total_pairs}")
    logger.info(f"Successful: {successful_pairs}")
    logger.info(f"Failed: {len(failed_pairs)}")
    
    if failed_pairs:
        logger.error("Failed language pairs:")
        for pair in failed_pairs:
            logger.error(f"  - {pair}")
    else:
        logger.info("All language pairs processed successfully!")
    
    logger.info("=" * 50)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and process NLLB datasets from Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--directory", "-d", type=str, required=True, help="Directory to save downloaded raw data")
    parser.add_argument("--dataset", "-ds", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--logfile", "-l", type=str, default="nllb_processor.log", help="Path to the log file")
    parser.add_argument("--loglevel", "-ll", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    parser.add_argument("--num_proc", "-np", type=int, default=1, help="Number of processes")
    parser.add_argument("--pairslist", "-pl", type=str, default=None, 
                       help="Comma-separated list of language pairs")
    parser.add_argument("--force_redownload", "-fr", action="store_true", 
                       help="Force reprocessing even if output exists")
    
    return parser

def main() -> None:
    """Data source: https://huggingface.co/datasets/allenai/nllb"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    
    # Setup logging and environment
    logger = setup_logging(args)
    
    # Run processing
    failed_pairs = run_processing(args, logger)
    
    # Get total pairs count for summary
    pairs = get_language_pairs(args, logger)
    total_pairs = len(pairs)
    
    # Final summary
    log_final_summary(total_pairs, failed_pairs, logger)

if __name__ == "__main__":
    main()
