#!/usr/bin/env python3
"""
Script to convert and save SONAR Speech2Text model with optional inference test.
"""
import argparse
import os
import sys

import torch
from transformers import AutoFeatureExtractor, AutoTokenizer

# Add current directory to path
sys.path.append(os.getcwd())

from open_sonar.speech.models.modeling import SONARForSpeech2Text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert SONAR Speech2Text model and optionally test inference"
    )
    parser.add_argument(
        "--encoder_id",
        type=str,
        default="facebook/w2v-bert-2.0",
        help="HuggingFace encoder model ID (default: facebook/w2v-bert-2.0)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="tutur90/SONAR-Text-to-Text",
        help="HuggingFace model ID to convert, set your own checkpoint if you want to (default: tutur90/SONAR-Text-to-Text)"
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default="open_sonar/speech/models/pretrained/sonar_speech",
        help="Output path to save the converted model (default: open_sonar/speech/models/pretrained/sonar_speech)"
    )
    parser.add_argument(
        "--test_audio_length",
        type=int,
        default=10000,
        help="Length of dummy audio for testing (default: 10000)"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip the inference test after model conversion"
    )
    parser.add_argument(
        "--show_params",
        action="store_true",
        help="Display the number of model parameters"
    )
    return parser.parse_args()


def convert_and_save_model(model_id, encoder_id, model_output_path):
    """Convert SONAR Speech2Text model and save it."""
    print(f"Converting model from {model_id} with encoder {encoder_id}...")
    model = SONARForSpeech2Text.from_sonar_w2v_pretrained(model_id, encoder_id)
    
    # Set processor class
    model.config.processor_class = "Wav2Vec2Processor"
    print("Model config:", model.config)
    
    print(f"Saving model to {model_output_path}...")
    model.save_pretrained(model_output_path)
    
    print(f"Saving feature extractor to {model_output_path}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
    feature_extractor.save_pretrained(model_output_path)
    
    print(f"Saving tokenizer to {model_output_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(model_output_path)
    
    print("Model, feature extractor, and tokenizer saved successfully!")


def count_parameters(model):
    """Count and display the number of model parameters."""
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {num_params:,}")
    return num_params


def test_inference(model_output_path, test_audio_length, max_length=128, num_beams=5):
    """Load the saved model and test inference with dummy audio."""
    print(f"\nLoading model from {model_output_path}...")
    model = SONARForSpeech2Text.from_pretrained(model_output_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_output_path)
    
    print(f"Running inference on dummy audio (length: {test_audio_length})...")
    # Create dummy audio input
    dummy_audio = torch.ones((1, test_audio_length))
    inputs = feature_extractor(dummy_audio, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams
        )
    
    print(f"Generated output IDs: {out}")
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Decoded output: {decoded}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Convert and save the model
    convert_and_save_model(args.model_id, args.encoder_id, args.model_output_path)
    
    # Show parameter count if requested
    if args.show_params:
        model = SONARForSpeech2Text.from_pretrained(args.model_output_path)
        count_parameters(model)
    
    # Test inference if not skipped
    if not args.skip_inference:
        test_inference(
            args.model_output_path,
            args.test_audio_length,
        )


if __name__ == "__main__":
    main()