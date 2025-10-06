#!/usr/bin/env python3
"""
Script to convert and save NLLB model to SONAR format with optional inference test.
"""
import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

# Add current directory to path
sys.path.append(os.getcwd())

from open_sonar.text.models.modeling_sonar import SONARForText2Text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert NLLB model to SONAR format and optionally test inference"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/nllb-200-1.3B",
        help="HuggingFace model ID to convert (default: facebook/nllb-200-1.3B)"
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default="open_sonar/text/models/pretrained/nllb_1.3B",
        help="Output path to save the converted model (default: open_sonar/text/models/pretrained/nllb_1.3B)"
    )
    parser.add_argument(
        "--test_text",
        type=str,
        default="Hello, my dog is cute",
        help="Text to use for inference test (default: 'Hello, my dog is cute')"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum length for generation (default: 30)"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip the inference test after model conversion"
    )
    return parser.parse_args()


def convert_and_save_model(model_id, model_output_path):
    """Convert NLLB model to SONAR format and save it."""
    print(f"Converting model from {model_id}...")
    model = SONARForText2Text.from_m2m100_pretrained(model_id)
    
    print(f"Saving model to {model_output_path}...")
    model.save_pretrained(model_output_path)
    
    print(f"Saving tokenizer to {model_output_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(model_output_path)
    
    print("Model and tokenizer saved successfully!")


def test_inference(model_output_path, test_text, max_length):
    """Load the saved model and test inference."""
    print(f"\nLoading model from {model_output_path}...")
    model = SONARForText2Text.from_pretrained(model_output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_output_path)
    
    print(f"Running inference on: '{test_text}'")
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length
        )
    
    print(f"Generated output IDs: {out}")
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Decoded output: {decoded}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Convert and save the model
    convert_and_save_model(args.model_id, args.model_output_path)
    
    # Test inference if not skipped
    if not args.skip_inference:
        test_inference(args.model_output_path, args.test_text, args.max_length)


if __name__ == "__main__":
    main()