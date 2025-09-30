#!/usr/bin/env python3
"""
Script to map datasets using Hugging Face models
Supports various tasks like text classification, sentiment analysis, and text generation
"""

import os

from datasets import Dataset, load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
import torch
from tqdm import tqdm
import argparse
import logging

import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


from sonar.tokenizerV2 import NllbTokenizerFast
from transformers import Wav2Vec2Processor, AutoFeatureExtractor, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Map dataset with Hugging Face model")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--data_path", required=True, help="Dataset path or name")
    parser.add_argument("--output_path", required=True, help="Path for output")
    parser.add_argument("--text_column", default="text", help="Name of text column")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes")
    parser.add_argument("--lang", default="en", help="Language code")

    args = parser.parse_args()
    
    try:

        print("running on", args.num_proc, " processes")
        
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
        
        # Load dataset
        dataset = load_dataset(args.data_path, args.lang, num_proc=args.num_proc)
        
        print(dataset)

        print("Dataset loaded")

        # Initialize mapper
        def prepare_eval_dataset(batch):
            # process audio
            sample = batch["audio"]
            inputs = feature_extractor(
                sample["array"], sampling_rate=sample["sampling_rate"]
            )
            # process audio length
            batch["input_features"] = inputs.get("input_features")[0]
            batch["lengths"] = len(sample["array"])

            batch["attention_mask"] = inputs.get("attention_mask")[0]

            # process targets
            input_str = batch[args.text_column]
            batch["labels"] = tokenizer(input_str).input_ids
            return batch
        
        # Process dataset
        processed_dataset = dataset.map(
            prepare_eval_dataset,
            remove_columns=dataset["train"].column_names,
            num_proc=args.num_proc,
            desc="Preparing eval dataset",
        ).sort("length")

        processed_dataset.save_to_disk(args.output_path)

        
        logger.info("Processing completed successfully!")
        
        # Print sample results
        print("\nSample results:")
        # print(processed_dataset[:3])
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Check if running with command line arguments
    main()
