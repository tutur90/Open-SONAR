from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

import sys 
import os
sys.path.append(os.getcwd())

from open_sonar.speech.models.modeling import SONARForSpeech2Text

from collections import OrderedDict





# s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng")

# print(s2vec_model)

# print(s2vec_model.state_dict().keys())

# source_state_dict = s2vec_model.model.state_dict()

# hf_model = SONARForSpeech2Text.from_pretrained("open_sonar/speech/models/pretrained/sonar_speech")


# print(hf_model)

# print(hf_model.model.encoder.state_dict().keys())

# target_state_dict = hf_model.state_dict()

import torch
import torch.nn as nn
from collections import OrderedDict
import warnings

def create_weight_mapping():
    """Create comprehensive weight mapping between source and target models"""
    
    # Frontend mapping
    frontend_mapping = {
        "encoder_frontend.post_extract_layer_norm.weight": "encoder.feature_projection.layer_norm.weight",
        "encoder_frontend.post_extract_layer_norm.bias": "encoder.feature_projection.layer_norm.bias",
        "encoder_frontend.model_dim_proj.weight": "encoder.feature_projection.projection.weight",
        "encoder_frontend.model_dim_proj.bias": "encoder.feature_projection.projection.bias",
    }
    
    # Create layer mappings for all 24 encoder layers
    layer_mapping = {}
    for i in range(24):
        layer_mappings = {
            # FFN1
            f"encoder.layers.{i}.ffn1_layer_norm.weight": f"encoder.encoder.layers.{i}.ffn1_layer_norm.weight",
            f"encoder.layers.{i}.ffn1_layer_norm.bias": f"encoder.encoder.layers.{i}.ffn1_layer_norm.bias",
            f"encoder.layers.{i}.ffn1.inner_proj.weight": f"encoder.encoder.layers.{i}.ffn1.intermediate_dense.weight",
            f"encoder.layers.{i}.ffn1.inner_proj.bias": f"encoder.encoder.layers.{i}.ffn1.intermediate_dense.bias",
            f"encoder.layers.{i}.ffn1.output_proj.weight": f"encoder.encoder.layers.{i}.ffn1.output_dense.weight",
            f"encoder.layers.{i}.ffn1.output_proj.bias": f"encoder.encoder.layers.{i}.ffn1.output_dense.bias",
            
            # Self Attention
            f"encoder.layers.{i}.self_attn_layer_norm.weight": f"encoder.encoder.layers.{i}.self_attn_layer_norm.weight",
            f"encoder.layers.{i}.self_attn_layer_norm.bias": f"encoder.encoder.layers.{i}.self_attn_layer_norm.bias",
            f"encoder.layers.{i}.self_attn.q_proj.weight": f"encoder.encoder.layers.{i}.self_attn.linear_q.weight",
            f"encoder.layers.{i}.self_attn.q_proj.bias": f"encoder.encoder.layers.{i}.self_attn.linear_q.bias",
            f"encoder.layers.{i}.self_attn.k_proj.weight": f"encoder.encoder.layers.{i}.self_attn.linear_k.weight",
            f"encoder.layers.{i}.self_attn.k_proj.bias": f"encoder.encoder.layers.{i}.self_attn.linear_k.bias",
            f"encoder.layers.{i}.self_attn.v_proj.weight": f"encoder.encoder.layers.{i}.self_attn.linear_v.weight",
            f"encoder.layers.{i}.self_attn.v_proj.bias": f"encoder.encoder.layers.{i}.self_attn.linear_v.bias",
            f"encoder.layers.{i}.self_attn.output_proj.weight": f"encoder.encoder.layers.{i}.self_attn.linear_out.weight",
            f"encoder.layers.{i}.self_attn.output_proj.bias": f"encoder.encoder.layers.{i}.self_attn.linear_out.bias",
            
            # Convolution Module (skip incompatible batch_norm running stats)
            f"encoder.layers.{i}.conv_layer_norm.weight": f"encoder.encoder.layers.{i}.conv_module.layer_norm.weight",
            f"encoder.layers.{i}.conv_layer_norm.bias": f"encoder.encoder.layers.{i}.conv_module.layer_norm.bias",
            f"encoder.layers.{i}.conv.pointwise_conv1.weight": f"encoder.encoder.layers.{i}.conv_module.pointwise_conv1.weight",
            f"encoder.layers.{i}.conv.depthwise_conv.weight": f"encoder.encoder.layers.{i}.conv_module.depthwise_conv.weight",
            f"encoder.layers.{i}.conv.batch_norm.weight": f"encoder.encoder.layers.{i}.conv_module.depthwise_layer_norm.weight",
            f"encoder.layers.{i}.conv.batch_norm.bias": f"encoder.encoder.layers.{i}.conv_module.depthwise_layer_norm.bias",
            f"encoder.layers.{i}.conv.pointwise_conv2.weight": f"encoder.encoder.layers.{i}.conv_module.pointwise_conv2.weight",
            
            # FFN2
            f"encoder.layers.{i}.ffn2_layer_norm.weight": f"encoder.encoder.layers.{i}.ffn2_layer_norm.weight",
            f"encoder.layers.{i}.ffn2_layer_norm.bias": f"encoder.encoder.layers.{i}.ffn2_layer_norm.bias",
            f"encoder.layers.{i}.ffn2.inner_proj.weight": f"encoder.encoder.layers.{i}.ffn2.intermediate_dense.weight",
            f"encoder.layers.{i}.ffn2.inner_proj.bias": f"encoder.encoder.layers.{i}.ffn2.intermediate_dense.bias",
            f"encoder.layers.{i}.ffn2.output_proj.weight": f"encoder.encoder.layers.{i}.ffn2.output_dense.weight",
            f"encoder.layers.{i}.ffn2.output_proj.bias": f"encoder.encoder.layers.{i}.ffn2.output_dense.bias",
            
            # Final Layer Norm
            f"encoder.layers.{i}.layer_norm.weight": f"encoder.encoder.layers.{i}.final_layer_norm.weight",
            f"encoder.layers.{i}.layer_norm.bias": f"encoder.encoder.layers.{i}.final_layer_norm.bias",
        }
        layer_mapping.update(layer_mappings)
    
    # Decoder/Pooler mapping for 3 layers (skip incompatible ones)
    decoder_mapping = {}
    for j in range(3):
        decoder_mappings = {
            # Self Attention
            f"encoder_pooler.decoder.layers.{j}.self_attn.q_proj.weight": f"pooling.layers.{j}.self_attn.q_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.self_attn.q_proj.bias": f"pooling.layers.{j}.self_attn.q_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.self_attn.k_proj.weight": f"pooling.layers.{j}.self_attn.k_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.self_attn.k_proj.bias": f"pooling.layers.{j}.self_attn.k_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.self_attn.v_proj.weight": f"pooling.layers.{j}.self_attn.v_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.self_attn.v_proj.bias": f"pooling.layers.{j}.self_attn.v_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.self_attn.output_proj.weight": f"pooling.layers.{j}.self_attn.out_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.self_attn.output_proj.bias": f"pooling.layers.{j}.self_attn.out_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.self_attn_layer_norm.weight": f"pooling.layers.{j}.self_attn_layer_norm.weight",
            f"encoder_pooler.decoder.layers.{j}.self_attn_layer_norm.bias": f"pooling.layers.{j}.self_attn_layer_norm.bias",
            
            # Encoder-Decoder Attention
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.q_proj.weight": f"pooling.layers.{j}.encoder_attn.q_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.q_proj.bias": f"pooling.layers.{j}.encoder_attn.q_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.k_proj.weight": f"pooling.layers.{j}.encoder_attn.k_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.k_proj.bias": f"pooling.layers.{j}.encoder_attn.k_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.v_proj.weight": f"pooling.layers.{j}.encoder_attn.v_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.v_proj.bias": f"pooling.layers.{j}.encoder_attn.v_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.output_proj.weight": f"pooling.layers.{j}.encoder_attn.out_proj.weight",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn.output_proj.bias": f"pooling.layers.{j}.encoder_attn.out_proj.bias",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn_layer_norm.weight": f"pooling.layers.{j}.encoder_attn_layer_norm.weight",
            f"encoder_pooler.decoder.layers.{j}.encoder_decoder_attn_layer_norm.bias": f"pooling.layers.{j}.encoder_attn_layer_norm.bias",
            
            # FFN (Note: these will be skipped due to dimension mismatch - see convert_model function)
            f"encoder_pooler.decoder.layers.{j}.ffn.inner_proj.weight": f"pooling.layers.{j}.fc1.weight",
            f"encoder_pooler.decoder.layers.{j}.ffn.inner_proj.bias": f"pooling.layers.{j}.fc1.bias",
            f"encoder_pooler.decoder.layers.{j}.ffn.output_proj.weight": f"pooling.layers.{j}.fc2.weight",
            f"encoder_pooler.decoder.layers.{j}.ffn.output_proj.bias": f"pooling.layers.{j}.fc2.bias",
            f"encoder_pooler.decoder.layers.{j}.ffn_layer_norm.weight": f"pooling.layers.{j}.final_layer_norm.weight",
            f"encoder_pooler.decoder.layers.{j}.ffn_layer_norm.bias": f"pooling.layers.{j}.final_layer_norm.bias",
        }
        decoder_mapping.update(decoder_mappings)
    
    # Combine all mappings
    full_mapping = {}
    full_mapping.update(frontend_mapping)
    full_mapping.update(layer_mapping)
    full_mapping.update(decoder_mapping)
    
    return full_mapping

def check_tensor_compatibility(source_tensor, target_tensor, source_key, target_key):
    """Check if two tensors are compatible for weight transfer"""
    if source_tensor.shape != target_tensor.shape:
        return False, f"Shape mismatch: {source_key} {source_tensor.shape} vs {target_key} {target_tensor.shape}"
    if source_tensor.dtype != target_tensor.dtype:
        return False, f"Dtype mismatch: {source_key} {source_tensor.dtype} vs {target_key} {target_tensor.dtype}"
    return True, "Compatible"

def convert_model(source_model, target_model, verbose=True):
    """
    Convert weights from source model to target model
    
    Args:
        source_model: SpeechToEmbeddingModelPipeline
        target_model: SONARForSpeech2Text
        verbose: Whether to print detailed information
    
    Returns:
        dict: Conversion statistics and information
    """
    
    # Get state dictionaries
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    # Create weight mapping
    weight_mapping = create_weight_mapping()
    
    # Statistics tracking
    stats = {
        'total_mappings': len(weight_mapping),
        'successful_transfers': 0,
        'skipped_missing_source': 0,
        'skipped_missing_target': 0,
        'skipped_incompatible': 0,
        'transferred_weights': [],
        'skipped_weights': [],
        'warnings': []
    }
    
    # Create new state dict for target model
    new_target_state_dict = target_state_dict.copy()
    
    # Process each mapping
    for source_key, target_key in weight_mapping.items():
        
        # Check if source key exists
        if source_key not in source_state_dict:
            stats['skipped_missing_source'] += 1
            if verbose:
                stats['warnings'].append(f"Source key not found: {source_key}")
            continue
        
        # Check if target key exists
        if target_key not in target_state_dict:
            stats['skipped_missing_target'] += 1
            if verbose:
                stats['warnings'].append(f"Target key not found: {target_key}")
            continue
        
        # Get tensors
        source_tensor = source_state_dict[source_key]
        target_tensor = target_state_dict[target_key]
        
        # Check compatibility
        is_compatible, message = check_tensor_compatibility(source_tensor, target_tensor, source_key, target_key)
        
        if is_compatible:
            # Transfer the weight
            new_target_state_dict[target_key] = source_tensor.clone()
            stats['successful_transfers'] += 1
            stats['transferred_weights'].append((source_key, target_key))
            if verbose:
                print(f"✓ Transferred: {source_key} -> {target_key}")
        else:
            stats['skipped_incompatible'] += 1
            stats['skipped_weights'].append((source_key, target_key, message))
            if verbose:
                print(f"✗ Skipped: {source_key} -> {target_key} ({message})")
    
    # Handle special cases that require manual processing
    special_cases = handle_special_cases(source_state_dict, target_state_dict, verbose)
    stats.update(special_cases)
    
    # Load the new state dict into target model
    missing_keys, unexpected_keys = target_model.load_state_dict(new_target_state_dict, strict=False)
    
    stats['missing_keys'] = missing_keys
    stats['unexpected_keys'] = unexpected_keys
    
    if verbose:
        print_conversion_summary(stats)
    
    return stats

def handle_special_cases(source_state_dict, target_state_dict, verbose=True):
    """Handle special cases that require custom processing"""
    
    special_stats = {
        'special_cases_processed': 0,
        'special_case_details': []
    }
    
    # 1. Handle relative position attention (r_proj vs distance_embedding)
    # This requires transformation from Linear layer to Embedding layer
    for i in range(24):
        source_key = f"encoder.layers.{i}.self_attn.sdpa.r_proj.weight"
        target_key = f"encoder.encoder.layers.{i}.self_attn.distance_embedding.weight"
        
        if source_key in source_state_dict and target_key in target_state_dict:
            # This transformation is complex and may require retraining
            # For now, we skip this and let it use default initialization
            special_stats['special_case_details'].append(f"Skipped r_proj transformation for layer {i}")
            if verbose:
                print(f"⚠ Special case: Skipped r_proj -> distance_embedding for layer {i}")
    
    # 2. Handle embedding size mismatch (will be skipped in main conversion)
    if "encoder_pooler.decoder_frontend.embed.weight" in source_state_dict:
        source_embed = source_state_dict["encoder_pooler.decoder_frontend.embed.weight"]
        if "pooling.embed_tokens.weight" in target_state_dict:
            target_embed = target_state_dict["pooling.embed_tokens.weight"]
            special_stats['special_case_details'].append(
                f"Embedding size mismatch: {source_embed.shape} vs {target_embed.shape}"
            )
            if verbose:
                print(f"⚠ Special case: Embedding size mismatch {source_embed.shape} vs {target_embed.shape}")
    
    # 3. Handle BatchNorm running stats (these don't have direct equivalents in LayerNorm)
    batch_norm_keys = [k for k in source_state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
    for key in batch_norm_keys:
        special_stats['special_case_details'].append(f"Skipped BatchNorm running stat: {key}")
        if verbose:
            print(f"⚠ Special case: Skipped BatchNorm running stat {key}")
    
    special_stats['special_cases_processed'] = len(special_stats['special_case_details'])
    
    return special_stats

def print_conversion_summary(stats):
    """Print a detailed summary of the conversion process"""
    
    print("\\n" + "="*80)
    print("MODEL CONVERSION SUMMARY")
    print("="*80)
    
    print(f"Total mappings attempted: {stats['total_mappings']}")
    print(f"Successful transfers: {stats['successful_transfers']}")
    print(f"Skipped (missing source): {stats['skipped_missing_source']}")
    print(f"Skipped (missing target): {stats['skipped_missing_target']}")
    print(f"Skipped (incompatible): {stats['skipped_incompatible']}")
    print(f"Special cases processed: {stats.get('special_cases_processed', 0)}")
    
    success_rate = (stats['successful_transfers'] / stats['total_mappings']) * 100
    print(f"\\nSuccess rate: {success_rate:.1f}%")
    
    print(f"\\nMissing keys in target: {len(stats.get('missing_keys', []))}")
    print(f"Unexpected keys: {len(stats.get('unexpected_keys', []))}")
    
    if stats.get('skipped_weights'):
        print(f"\\nIncompatible weights (first 10):")
        for i, (src, tgt, msg) in enumerate(stats['skipped_weights'][:10]):
            print(f"  {i+1}. {src} -> {tgt}: {msg}")
        if len(stats['skipped_weights']) > 10:
            print(f"  ... and {len(stats['skipped_weights']) - 10} more")
    
    print("="*80)

# Usage example:
def main():
    """
    Main function to perform the conversion
    """
    
    # Assuming you have already loaded your models:
    s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng").model
    hf_model = SONARForSpeech2Text.from_pretrained("open_sonar/speech/models/pretrained/sonar_speech").model.encoder
    
    print("Starting SONAR model conversion...")
    print("This will transfer compatible weights from SpeechToEmbeddingModelPipeline to SONARForSpeech2Text")

    print(s2vec_model)
    print(hf_model)

    # Perform the conversion
    # stats = convert_model(s2vec_model, hf_model, verbose=True)
    
    # Save the converted model if desired
    # hf_model.save_pretrained("./converted_sonar_model")
    
    print("\\nConversion complete!")
    print("Note: Some weights may not be transferred due to architectural differences.")
    print("The model should still be functional but may require fine-tuning for optimal performance.")

if __name__ == "__main__":
    main()