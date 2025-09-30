from typing import Optional, Tuple, Union, List
import math
# import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    M2M100Config,
    M2M100PreTrainedModel,
    GenerationConfig,)

from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100EncoderLayer,
    M2M100SinusoidalPositionalEmbedding,
    M2M100Decoder,
    M2M100Encoder,
    M2M100ScaledWordEmbedding,
    M2M100ForConditionalGeneration,
    M2M100Model)

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.models.m2m_100.modeling_m2m_100 import _prepare_4d_attention_mask, _prepare_4d_attention_mask_for_sdpa, shift_tokens_right
from transformers.models.m2m_100.modeling_m2m_100 import logger

from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache, EncoderDecoderCache
# from sonar.models.model import SONARForConditionalGeneration


_CONFIG_FOR_DOC = "M2M100Config"
_CHECKPOINT_FOR_DOC = "facebook/m2m100_418M"

torch.set_printoptions(threshold=10_000)

class CorrectedSONAREncoder(M2M100Encoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`M2M100EncoderLayer`].

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        
        self.correcting_mlp = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.lang_weighting = nn.Embedding(
            config.vocab_size,
            1,
        )
        
        self.correcting_embedding = nn.Parameter(
            torch.zeros(1, 1),
            requires_grad=True,
        )
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pool: Optional[str] = "mean",
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        
        if attention_mask is not None:
        
            attention_mask_saved = attention_mask.unsqueeze(-1)
            
            seq_length = attention_mask_saved.sum(1, keepdim=True)
            

            lang_weights = self.lang_weighting(input_ids[:, 0].unsqueeze(1))
            

        else:
            print("Attention mask is None, you will probably get big problems :/")
            
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = output.last_hidden_state
        
        output.unpool_hidden_states = hidden_states
        

        if attention_mask_saved is not None:
            
            output.last_hidden_state = (hidden_states * attention_mask_saved).sum(1, keepdim=True) / seq_length
            


            corrected_lengths = self.correcting_mlp(seq_length.float())
            

            corr =  corrected_lengths * lang_weights
            


            output.last_hidden_state = output.last_hidden_state * self.correcting_embedding + corr


        return output
    
    
class CorrectedSONARModel(M2M100Model):
    """SONAR model based on M2M100."""

    
    def __init__(self, config: M2M100Config):
        super().__init__(config)
        # Replace with your custom encoder/decoder
        self.encoder = CorrectedSONAREncoder(config, self.shared)
        

# class SONARForConditionalGeneration(SONARForConditionalGeneration):
#     """SONAR model for conditional generation based on M2M100."""

#     def __init__(self, config: M2M100Config):
#         super().__init__(config)
#         self.model = CorrectedSONARModel(config)