from hmac import new
from idna import decode
import sonar
import torch
from collections import OrderedDict
from transformers import M2M100ForConditionalGeneration
# from sonar.inference_pipelines.text import (
#     TextToTextModelPipeline,
# )

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union, cast

import fairseq2
import torch
from fairseq2.data import Collater, read_sequence
from fairseq2.data.text import read_text
from fairseq2.data.text.tokenizers import TextTokenizer, get_text_tokenizer_hub
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    Sampler,
    SamplingSeq2SeqGenerator,
    Seq2SeqGenerator,
)

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.generation.text import SequenceToTextConverter, TextTranslator
from fairseq2.typing import CPU, DataType, Device

from sonar.inference_pipelines.utils import add_progress_bar, extract_sequence_batch
from sonar.models.encoder_model import SonarEncoderModel
from sonar.models.sonar_text import (
    get_sonar_text_decoder_hub,
    get_sonar_text_encoder_hub,
)
from sonar.models.sonar_translation import SonarEncoderDecoderModel
from sonar.models.sonar_translation.model import DummyEncoderModel
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel


def convert_m2m100_to_sonar(hf_state_dict):
    encoder_state_dict = OrderedDict()

    # --- Shared embeddings ---
    encoder_state_dict["encoder_frontend.embed.weight"] = hf_state_dict["model.shared.weight"]
    # encoder_state_dict["decoder_frontend.embed.weight"] = hf_state_dict["model.shared.weight"]
    
    # print(hf_state_dict.keys())
    
    num_layers = 12

    for i in range(num_layers):
        prefix_sonar = f"encoder.layers.{i}"
        prefix_m2m = f"model.encoder.layers.{i}"

        # Self-attention
        encoder_state_dict[f"{prefix_sonar}.self_attn.q_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.q_proj.weight"]
        encoder_state_dict[f"{prefix_sonar}.self_attn.q_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.q_proj.bias"]
        encoder_state_dict[f"{prefix_sonar}.self_attn.k_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.k_proj.weight"]
        encoder_state_dict[f"{prefix_sonar}.self_attn.k_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.k_proj.bias"]
        encoder_state_dict[f"{prefix_sonar}.self_attn.v_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.v_proj.weight"]
        encoder_state_dict[f"{prefix_sonar}.self_attn.v_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.v_proj.bias"]
        encoder_state_dict[f"{prefix_sonar}.self_attn.output_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.out_proj.weight"]
        encoder_state_dict[f"{prefix_sonar}.self_attn.output_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.out_proj.bias"]

        # Layer norms
        encoder_state_dict[f"{prefix_sonar}.self_attn_layer_norm.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn_layer_norm.weight"]
        encoder_state_dict[f"{prefix_sonar}.self_attn_layer_norm.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn_layer_norm.bias"]

        encoder_state_dict[f"{prefix_sonar}.ffn_layer_norm.weight"] = hf_state_dict[f"{prefix_m2m}.final_layer_norm.weight"]
        encoder_state_dict[f"{prefix_sonar}.ffn_layer_norm.bias"]   = hf_state_dict[f"{prefix_m2m}.final_layer_norm.bias"]

        # Feed Forward
        encoder_state_dict[f"{prefix_sonar}.ffn.inner_proj.weight"] = hf_state_dict[f"{prefix_m2m}.fc1.weight"]
        encoder_state_dict[f"{prefix_sonar}.ffn.inner_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.fc1.bias"]
        encoder_state_dict[f"{prefix_sonar}.ffn.output_proj.weight"] = hf_state_dict[f"{prefix_m2m}.fc2.weight"]
        encoder_state_dict[f"{prefix_sonar}.ffn.output_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.fc2.bias"]

    # Encoder final layer norm
    encoder_state_dict["layer_norm.weight"] = hf_state_dict["model.encoder.layer_norm.weight"]
    encoder_state_dict["layer_norm.bias"]   = hf_state_dict["model.encoder.layer_norm.bias"]
    # --- Decoder ---
    
    decoder_state_dict = OrderedDict()
    decoder_state_dict["decoder_frontend.embed.weight"] = hf_state_dict["model.shared.weight"]
    # decoder_state_dict["decoder_frontend.embed.weight"] = hf_state_dict["model.shared
    
    # decoder_state_dict["decoder_frontend.pos_encoder.pe"] = hf_state_dict["model.decoder.embed_positions.weight"] if "model.decoder.embed_positions.weight" in hf_state_dict else hf_state_dict["model.shared.weight"]

    for i in range(num_layers):
        prefix_sonar = f"decoder.layers.{i}"
        prefix_m2m = f"model.decoder.layers.{i}"

        # Self-attention
        decoder_state_dict[f"{prefix_sonar}.self_attn.q_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.q_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.self_attn.q_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.q_proj.bias"]
        decoder_state_dict[f"{prefix_sonar}.self_attn.k_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.k_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.self_attn.k_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.k_proj.bias"]
        decoder_state_dict[f"{prefix_sonar}.self_attn.v_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.v_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.self_attn.v_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.v_proj.bias"]
        decoder_state_dict[f"{prefix_sonar}.self_attn.output_proj.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn.out_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.self_attn.output_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn.out_proj.bias"]

        # Cross-attention
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.q_proj.weight"] = hf_state_dict[f"{prefix_m2m}.encoder_attn.q_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.q_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.encoder_attn.q_proj.bias"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.k_proj.weight"] = hf_state_dict[f"{prefix_m2m}.encoder_attn.k_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.k_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.encoder_attn.k_proj.bias"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.v_proj.weight"] = hf_state_dict[f"{prefix_m2m}.encoder_attn.v_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.v_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.encoder_attn.v_proj.bias"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.output_proj.weight"] = hf_state_dict[f"{prefix_m2m}.encoder_attn.out_proj.weight"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn.output_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.encoder_attn.out_proj.bias"]

        # Layer norms
        decoder_state_dict[f"{prefix_sonar}.self_attn_layer_norm.weight"] = hf_state_dict[f"{prefix_m2m}.self_attn_layer_norm.weight"]
        decoder_state_dict[f"{prefix_sonar}.self_attn_layer_norm.bias"]   = hf_state_dict[f"{prefix_m2m}.self_attn_layer_norm.bias"]

        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn_layer_norm.weight"] = hf_state_dict[f"{prefix_m2m}.encoder_attn_layer_norm.weight"]
        decoder_state_dict[f"{prefix_sonar}.encoder_decoder_attn_layer_norm.bias"]   = hf_state_dict[f"{prefix_m2m}.encoder_attn_layer_norm.bias"]

        decoder_state_dict[f"{prefix_sonar}.ffn_layer_norm.weight"] = hf_state_dict[f"{prefix_m2m}.final_layer_norm.weight"]
        decoder_state_dict[f"{prefix_sonar}.ffn_layer_norm.bias"]   = hf_state_dict[f"{prefix_m2m}.final_layer_norm.bias"]

        # Feed Forward
        decoder_state_dict[f"{prefix_sonar}.ffn.inner_proj.weight"] = hf_state_dict[f"{prefix_m2m}.fc1.weight"]
        decoder_state_dict[f"{prefix_sonar}.ffn.inner_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.fc1.bias"]
        decoder_state_dict[f"{prefix_sonar}.ffn.output_proj.weight"] = hf_state_dict[f"{prefix_m2m}.fc2.weight"]
        decoder_state_dict[f"{prefix_sonar}.ffn.output_proj.bias"]   = hf_state_dict[f"{prefix_m2m}.fc2.bias"]

    # Decoder final layer norm
    decoder_state_dict["decoder.layer_norm.weight"] = hf_state_dict["model.decoder.layer_norm.weight"]
    decoder_state_dict["decoder.layer_norm.bias"]   = hf_state_dict["model.decoder.layer_norm.bias"]

    # --- Final projection ---
    decoder_state_dict["final_proj.weight"] = hf_state_dict["lm_head.weight"]

    return encoder_state_dict, decoder_state_dict


if __name__ == "__main__":
    # Load Hugging Face checkpoint
    hf_model = M2M100ForConditionalGeneration.from_pretrained("checkpoints/sonar_mini")
    hf_state_dict = hf_model.state_dict()

    # # Convert
    sonar_sd = convert_m2m100_to_sonar(hf_state_dict)
    
    # sonar_sd = hf_state_dict
    

    torch.save({"model": sonar_sd[0],
                "fs2": True}, "encoder_converted.pt")
    
    torch.save({"model": sonar_sd[1],
                "fs2": True}, "decoder_converted.pt")

    print(hf_model)

    # register_nllb_configs()
    # model_hub = ModelHub(
    #     model=TransformerModel,

    # )
    
    sonar_model = TextToTextModelPipeline(encoder="text_sonar_basic_encoder",
                                    decoder="text_sonar_basic_decoder",
                                    tokenizer="text_sonar_basic_encoder").model

    print(sonar_model)
    
    # print(hf_model)
    
    # sonar_model.load_state_dict(sonar_sd)

    # Save converted checkpoint
    # torch.save(sonar_sd, "sonar_converted.pt")

    print("✅ Conversion done. Saved to sonar_converted.pt")
