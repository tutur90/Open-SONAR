#!/usr/bin/env python3
"""
Script to map datasets using Hugging Face models
Supports various tasks like text classification, sentiment analysis, and text generation
"""

import os
from pyexpat import model
import datasets
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoFeatureExtractor,
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


from models.model import SONAREncoder


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetMapper:
    def __init__(self, model_name, task_type="embed", device=None):
        """
        Initialize the dataset mapper with a Hugging Face model
        
        Args:
            model_name (str): Name of the HF model
            task_type (str): Type of task ('classification', 'sentiment', 'generation')
            device (str): Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.task_type = task_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing model {model_name} on {self.device}")
        
        # Initialize model and tokenizer based on task type
        if task_type in ["classification", "sentiment"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline(
                "text-classification", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        elif task_type == "generation":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        elif task_type == "embed":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = SONAREncoder.from_pretrained(model_name)
            self.pipeline = pipeline(
                "feature-extraction",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def process_batch(self, batch, text_column, batch_size=32):
        """
        Process a batch of texts with the model
        
        Args:
            batch (dict): Batch from dataset
            text_column (str): Name of the text column
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Processed batch with predictions
        """
        texts = batch[text_column]
        
        if self.task_type in ["classification", "sentiment"]:
            # Get predictions
            predictions = self.pipeline(texts, batch_size=batch_size)
            
            # Extract labels and scores
            labels = [pred['label'] for pred in predictions]
            scores = [pred['score'] for pred in predictions]
            
            batch['predicted_label'] = labels
            batch['confidence_score'] = scores
            
        elif self.task_type == "generation":
            # Generate text
            generated = self.pipeline(
                texts, 
                max_length=150, 
                num_return_sequences=1,
                batch_size=batch_size,
                truncation=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_texts = []
            for i, gen in enumerate(generated):
                if isinstance(gen, list):
                    generated_texts.append(gen[0]['generated_text'])
                else:
                    generated_texts.append(gen['generated_text'])
            
            batch['generated_text'] = generated_texts
            
        elif self.task_type == "embed":
            embeddings = self.pipeline(texts, batch_size=batch_size)
            batch['embeddings'] = [emb[0] for emb in embeddings]

        return batch
    
    def map_dataset(self, dataset: Dataset, text_column="text", batch_size=32, num_proc=1):
        """
        Map the entire dataset using the model
        
        Args:
            dataset (Dataset): Input dataset
            text_column (str): Name of the text column
            batch_size (int): Batch size for processing
            num_proc (int): Number of processes for mapping
            
        Returns:
            Dataset: Mapped dataset with predictions
        """
        logger.info(f"Processing dataset with {len(dataset)} examples")
        
        # Map the dataset
        mapped_dataset = dataset.map(
            lambda batch: self.process_batch(batch, text_column, batch_size),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Processing with model",
            load_from_cache_file=True
        )
        
        return mapped_dataset

def save_results(dataset, output_path, output_format="csv"):
    """
    Save processed results
    
    Args:
        dataset (Dataset): Processed dataset
        output_path (str): Output path
        output_format (str): Output format ('csv', 'json', 'parquet')
    """
    logger.info(f"Saving results to {output_path}")
    
    
    
    if output_format == "csv":
        df = dataset.to_pandas()
        df.to_csv(output_path, index=False)
    elif output_format == "json":
        df = dataset.to_pandas()
        df.to_json(output_path, orient="records", indent=2)
    elif output_format == "parquet":
        df = dataset.to_pandas()
        df.to_parquet(output_path, index=False)
    elif output_format == "arrow":
        dataset.save_to_disk(output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def main():
    parser = argparse.ArgumentParser(description="Map dataset with Hugging Face model")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--encoder_name", help="HuggingFace encoder model name for speech models")
    parser.add_argument("--data_path", required=True, help="Path to input data")
    parser.add_argument("--output_path", required=True, help="Path for output")
    parser.add_argument("--text_column", default="text", help="Name of text column")
    parser.add_argument("--task_type", choices=["embed", "classification", "sentiment", "generation"], 
                       default="embed", help="Type of task")
    parser.add_argument("--data_format", choices=["csv", "json", "parquet", "hf_dataset"], 
                       default="arrow", help="Input data format")
    parser.add_argument("--output_format", choices=["arrow", "csv", "json", "parquet"], 
                       default="csv", help="Output format")
    parser.add_argument("--batch_size", type=int, default=
                        1024, help="Batch size")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes")
    parser.add_argument("--lang", default="en", help="Language code")

    args = parser.parse_args()
    
    try:

        print("running on", args.num_proc, " processes")
        # Load dataset
        dataset = load_dataset(args.data_path, args.lang, split="train", num_proc=args.num_proc)

        print("Dataset loaded")

        # Initialize mapper
        mapper = DatasetMapper(args.encoder_name, args.task_type)
        
        # Process dataset
        # processed_dataset = mapper.map_dataset(
        #     dataset, 
        #     args.text_column, 
        #     args.batch_size, 
        #     1
        # )
        
        processed_dataset = load_from_disk("data/datasets/old/common_voice_17")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
        
        dataset_sampling_rate = next(iter(processed_dataset))["audio"]["sampling_rate"]
        if dataset_sampling_rate != feature_extractor.sampling_rate:
            processed_dataset = processed_dataset.cast_column(
                "audio", datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        
                # Initialize mapper
        def prepare_dataset(batch):
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
        
        processed_dataset = processed_dataset.map(
            prepare_dataset,
            remove_columns=list(set(dataset.column_names) - {"target_embeddings", "input_features", "attention_mask", "labels", "length"}),
            num_proc=args.num_proc,
            desc="preprocess dataset",
            load_from_cache_file=True,
        )
        
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
