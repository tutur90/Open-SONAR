from pathlib import Path
import datasets
import argparse
import os

import sys
import polars as pl

# sys.path.append(Path(__file__).parent.parent.parent.as_posix())

# from sonar.tokenizerV2 import NllbTokenizerFast

# tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-1.3B")


def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument("--input_files", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to use for processing")
    return parser.parse_args()



if __name__ == "__main__":
   args = parse_args()

   input_files = Path(args.input_files)
   output_path = Path(args.output_path)


   dataset = datasets.load_dataset("parquet", data_files={"train": str(input_files/"**/*.parquet")}, cache_dir=output_path, keep_in_memory=False, num_proc=args.num_proc).shuffle(seed=42, load_from_cache_file=False)
   print("Saving to disk")
   
   dataset.save_to_disk(output_path, num_proc=args.num_proc)
   # Process the dataset
   # Save the processed dataset