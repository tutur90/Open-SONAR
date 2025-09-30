import torch
from collections import OrderedDict
from transformers import M2M100ForConditionalGeneration

def convert_sonar_to_m2m100(sonar_state_dict, vocab_size=256206, model_dim=1024):
    new_state_dict = OrderedDict()

    # --- Shared embeddings ---
    new_state_dict["model.shared.weight"] = sonar_state_dict["encoder.encoder_frontend.embed.weight"]

    # --- Encoder ---
    new_state_dict["model.encoder.embed_tokens.weight"] = sonar_state_dict["encoder.encoder_frontend.embed.weight"]
    new_state_dict["model.encoder.embed_positions.weight"] = sonar_state_dict["encoder.encoder_frontend.pos_encoder.pe"]  # sinusoidal
    
    for i in range(24):
        prefix_sonar = f"encoder.encoder.layers.{i}"
        prefix_m2m = f"model.encoder.layers.{i}"

        # Self-attention
        new_state_dict[f"{prefix_m2m}.self_attn.q_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.q_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.q_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.q_proj.bias"]
        new_state_dict[f"{prefix_m2m}.self_attn.k_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.k_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.k_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.k_proj.bias"]
        new_state_dict[f"{prefix_m2m}.self_attn.v_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.v_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.v_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.v_proj.bias"]
        new_state_dict[f"{prefix_m2m}.self_attn.out_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.output_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.out_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.output_proj.bias"]

        # Layer norms
        new_state_dict[f"{prefix_m2m}.self_attn_layer_norm.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn_layer_norm.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn_layer_norm.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn_layer_norm.bias"]

        new_state_dict[f"{prefix_m2m}.final_layer_norm.weight"] = sonar_state_dict[f"{prefix_sonar}.ffn_layer_norm.weight"]
        new_state_dict[f"{prefix_m2m}.final_layer_norm.bias"]   = sonar_state_dict[f"{prefix_sonar}.ffn_layer_norm.bias"]

        # Feed Forward
        new_state_dict[f"{prefix_m2m}.fc1.weight"] = sonar_state_dict[f"{prefix_sonar}.ffn.inner_proj.weight"]
        new_state_dict[f"{prefix_m2m}.fc1.bias"]   = sonar_state_dict[f"{prefix_sonar}.ffn.inner_proj.bias"]
        new_state_dict[f"{prefix_m2m}.fc2.weight"] = sonar_state_dict[f"{prefix_sonar}.ffn.output_proj.weight"]
        new_state_dict[f"{prefix_m2m}.fc2.bias"]   = sonar_state_dict[f"{prefix_sonar}.ffn.output_proj.bias"]

    # Encoder final layer norm
    new_state_dict["model.encoder.layer_norm.weight"] = sonar_state_dict["encoder.layer_norm.weight"]
    new_state_dict["model.encoder.layer_norm.bias"]   = sonar_state_dict["encoder.layer_norm.bias"]

    # --- Decoder ---
    new_state_dict["model.decoder.embed_tokens.weight"] = sonar_state_dict["decoder.decoder_frontend.embed.weight"]
    new_state_dict["model.decoder.embed_positions.weight"] = sonar_state_dict["decoder.decoder_frontend.pos_encoder.pe"]

    for i in range(24):
        prefix_sonar = f"decoder.decoder.layers.{i}"
        prefix_m2m = f"model.decoder.layers.{i}"

        # Self-attention
        new_state_dict[f"{prefix_m2m}.self_attn.q_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.q_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.q_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.q_proj.bias"]
        new_state_dict[f"{prefix_m2m}.self_attn.k_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.k_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.k_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.k_proj.bias"]
        new_state_dict[f"{prefix_m2m}.self_attn.v_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.v_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.v_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.v_proj.bias"]
        new_state_dict[f"{prefix_m2m}.self_attn.out_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn.output_proj.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn.out_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn.output_proj.bias"]

        # Cross-attention
        new_state_dict[f"{prefix_m2m}.encoder_attn.q_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.q_proj.weight"]
        new_state_dict[f"{prefix_m2m}.encoder_attn.q_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.q_proj.bias"]
        new_state_dict[f"{prefix_m2m}.encoder_attn.k_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.k_proj.weight"]
        new_state_dict[f"{prefix_m2m}.encoder_attn.k_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.k_proj.bias"]
        new_state_dict[f"{prefix_m2m}.encoder_attn.v_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.v_proj.weight"]
        new_state_dict[f"{prefix_m2m}.encoder_attn.v_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.v_proj.bias"]
        new_state_dict[f"{prefix_m2m}.encoder_attn.out_proj.weight"] = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.output_proj.weight"]
        new_state_dict[f"{prefix_m2m}.encoder_attn.out_proj.bias"]   = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn.output_proj.bias"]

        # Layer norms
        new_state_dict[f"{prefix_m2m}.self_attn_layer_norm.weight"] = sonar_state_dict[f"{prefix_sonar}.self_attn_layer_norm.weight"]
        new_state_dict[f"{prefix_m2m}.self_attn_layer_norm.bias"]   = sonar_state_dict[f"{prefix_sonar}.self_attn_layer_norm.bias"]

        new_state_dict[f"{prefix_m2m}.encoder_attn_layer_norm.weight"] = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn_layer_norm.weight"]
        new_state_dict[f"{prefix_m2m}.encoder_attn_layer_norm.bias"]   = sonar_state_dict[f"{prefix_sonar}.encoder_decoder_attn_layer_norm.bias"]

        new_state_dict[f"{prefix_m2m}.final_layer_norm.weight"] = sonar_state_dict[f"{prefix_sonar}.ffn_layer_norm.weight"]
        new_state_dict[f"{prefix_m2m}.final_layer_norm.bias"]   = sonar_state_dict[f"{prefix_sonar}.ffn_layer_norm.bias"]

        # Feed Forward
        new_state_dict[f"{prefix_m2m}.fc1.weight"] = sonar_state_dict[f"{prefix_sonar}.ffn.inner_proj.weight"]
        new_state_dict[f"{prefix_m2m}.fc1.bias"]   = sonar_state_dict[f"{prefix_sonar}.ffn.inner_proj.bias"]
        new_state_dict[f"{prefix_m2m}.fc2.weight"] = sonar_state_dict[f"{prefix_sonar}.ffn.output_proj.weight"]
        new_state_dict[f"{prefix_m2m}.fc2.bias"]   = sonar_state_dict[f"{prefix_sonar}.ffn.output_proj.bias"]

    # Decoder final layer norm
    new_state_dict["model.decoder.layer_norm.weight"] = sonar_state_dict["decoder.decoder.layer_norm.weight"]
    new_state_dict["model.decoder.layer_norm.bias"]   = sonar_state_dict["decoder.decoder.layer_norm.bias"]

    # --- LM head ---
    new_state_dict["lm_head.weight"] = sonar_state_dict["decoder.final_proj.weight"]

    return new_state_dict


if __name__ == "__main__":
    # Load Sonar checkpoint
    sonar_ckpt = torch.load("sonar_model.pt", map_location="cpu")

    # Init Hugging Face model
    hf_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    # Convert weights
    new_sd = convert_sonar_to_m2m100(sonar_ckpt)

    # Load converted weights
    missing, unexpected = hf_model.load_state_dict(new_sd, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    torch.save(hf_model.state_dict(), "m2m100_converted.pt")
