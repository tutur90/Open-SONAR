from calendar import c
from dataclasses import dataclass
from json import encoder
import re
from shutil import copy
from typing import Optional, Tuple, Union, List
import math
# import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModel,
    M2M100Config,
    M2M100PreTrainedModel,
    )

from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Decoder,
    M2M100ScaledWordEmbedding,
    M2M100ForConditionalGeneration,
    shift_tokens_right)

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.m2m_100.modeling_m2m_100 import logger
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertModel, Wav2Vec2BertConfig

from transformers.generation import GenerationMixin


from transformers.utils import auto_docstring

import copy

# from open_sonar.speech.models.model import SONARConfig
from open_sonar.text.models.modeling_sonar  import SONARForText2Text, Pooling

from open_sonar.loss import CrossEntropyLoss





class SONARSpeechConfig(PretrainedConfig):
    r"""
    [`SONARSpeechConfig`] is the configuration class to store the configuration of a
    [`SONARSpeechModel`]. It is used to instantiate an Encoder Decoder model according to the specified
    arguments, defining the encoder and decoder configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

    >>> # Initializing a Wav2Vec2 & BERT style configuration
    >>> config_encoder = Wav2Vec2Config()
    >>> config_decoder = BertConfig()

    >>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a Wav2Vec2Bert model from a Wav2Vec2 & google-bert/bert-base-uncased style configurations
    >>> model = SpeechEncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = SpeechEncoderDecoderConfig.from_pretrained("my-model")
    >>> model = SpeechEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""

    model_type = "speech-encoder-decoder"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig, "pooling": AutoConfig}
    has_no_defaults_at_init = True
    is_encoder_decoder = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        encoder_config = kwargs.pop("encoder", {})
        encoder_model_type = encoder_config.pop("model_type", "w2v-bert-2.0")
        decoder_config = kwargs.pop("decoder", {})
        decoder_model_type = decoder_config.pop("model_type", "nllb-200-1.3B")
        pooling_config = kwargs.pop("pooling", {})
        pooling_model_type = pooling_config.pop("model_type", "m2m100")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.pooling = AutoConfig.for_model(pooling_model_type, **pooling_config)
        self.decoder_start_token_id = self.decoder.decoder_start_token_id
        self.eos_token_id = self.decoder.eos_token_id
        self.pad_token_id = self.decoder.pad_token_id
        self.vocab_size = self.decoder.vocab_size
        

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, pooling_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`SpeechEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`SpeechEncoderDecoderConfig`]: An instance of a configuration object
        """

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), pooling=pooling_config.to_dict(), **kwargs)


class AttentionPooling(M2M100Decoder):
    """
    Pooling layer for sequence representations.
    
    Supports multiple pooling strategies: mean, cls, last, max, and none.
    """

    def __init__(self, config: M2M100Config, shared: None, layers: int = 3):
        super().__init__(config, shared)

        self.bos_token_id = config.decoder_start_token_id
        
    def get_pooling_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get a learnable pooling token for 'cls' pooling."""
        return torch.tensor([self.bos_token_id]*batch_size, device=device, dtype=torch.int64)
            
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply pooling to hidden states.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len), values in {0, 1}
        
        Returns:
            Pooled tensor of shape (batch_size, 1, hidden_size) or (batch_size, seq_len, hidden_size) for none
        """
        
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        pooling_token = self.get_pooling_token(batch_size, device).unsqueeze(1)  # (batch_size, 1)

        decoder_outputs = super().forward(
            input_ids=pooling_token,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False
        )
        
        return decoder_outputs[0]  # (batch_size, 1, hidden_size)




class SONARSpeechEncoder(PreTrainedModel):
    
    

    """SONAR Speech Encoder.
    
    This class wraps a pre-trained speech encoder model.
    """
    config_class = SONARSpeechConfig

    def __init__(self, config: SONARSpeechConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config.encoder)
        self.encoder = Wav2Vec2BertModel(config.encoder)

        self.pooling = AttentionPooling(config = config.pooling, shared=embed_tokens)
        
        # self.pooling = Pooling()
        

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        r"""
        input_ids (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`):
            Input features.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.encoder.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.encoder.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.encoder.config.use_return_dict
        
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if return_dict:
            encoder_outputs.last_hidden_state = self.pooling(
                encoder_outputs.last_hidden_state,
                attention_mask=attention_mask
            )
        else:
            encoder_outputs[0] = self.pooling(
                encoder_outputs[0],
                attention_mask=attention_mask
            )
        
        return encoder_outputs

    @classmethod
    def from_w2v_bert_pretrained(cls, pretrained_model_name_or_path: str, pooling_config: dict, *model_args, pooling_layers: int = 0, **kwargs):
        r"""
        Instantiate a `SONARSpeechEncoder` model from a pretrained Wav2Vec2Bert model.
        """
        
        w2v_config = Wav2Vec2BertConfig.from_pretrained(pretrained_model_name_or_path)
        
        encoder = Wav2Vec2BertModel.from_pretrained(pretrained_model_name_or_path,
                                                    config=w2v_config
            )

        config = SONARSpeechConfig.from_encoder_decoder_configs(
            encoder_config=encoder.config,
            decoder_config=pooling_config,
            pooling_config=pooling_config
        )

        model = cls(config, *model_args, **kwargs)
        model.encoder.load_state_dict(encoder.state_dict(), strict=True)
        return model


class SONARModelS2T(M2M100PreTrainedModel):
    _tied_weights_keys = ["encoder.pooling.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: M2M100Config):
        super().__init__(config.decoder)

        padding_idx, vocab_size = config.decoder.pad_token_id, config.decoder.vocab_size
        embed_scale = math.sqrt(config.decoder.d_model) if config.decoder.scale_embedding else 1.0
        self.shared = M2M100ScaledWordEmbedding(vocab_size, config.decoder.d_model, padding_idx, embed_scale=embed_scale)

        self.encoder = SONARSpeechEncoder(config, embed_tokens=self.shared)
        self.decoder = M2M100Decoder(config.decoder)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            M2M100 uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if attention_mask is not None:
            if encoder_outputs[0].size(1) != 1 and encoder_outputs[0].dim() == 3:
                logger.warning_once(
                    f"Encoder is not pooled"
                )
                encoder_attention_mask = attention_mask
            else:
                encoder_attention_mask = attention_mask[:, :1]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _tie_weights(self):
        # if self.config.tie_word_embeddings:
        #     self._tie_or_clone_weights(self.encoder.pooling.embed_tokens, self.shared)
        #     self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
        pass
            
    def get_input_embeddings(self):
        return self.shared

    # def set_input_embeddings(self, value):
    #     self.shared = value
    #     self.encoder.embed_tokens = self.shared
    #     self.decoder.embed_tokens = self.shared


    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


        

class SONARForSpeech2Text(M2M100PreTrainedModel, GenerationMixin):
    """SONAR model for conditional generation tasks."""
    main_input_name = "input_features"
    
    config_class = SONARSpeechConfig

    _tied_weights_keys = ["lm_head.weight", 'model.shared.weight', 'model.encoder.pooling.embed_tokens.weight']


    def __init__(self, config):
        super().__init__(config)     
        
        config.is_encoder_decoder = True
        
        self.model = SONARModelS2T(config)
        
        self.lm_head = nn.Linear(config.decoder.d_model, config.decoder.vocab_size, bias=False)

        self.cross_entropy_loss = CrossEntropyLoss(
            label_smoothing=0.1,
            ignore_index=-100
        )
        
        self.mse_loss = nn.MSELoss()

        self.mse_ratio = 0.2
        
    @classmethod
    def from_model_encoder_pretrained(cls, pretrained_model_name_or_path: str, encoder_pretrained_model_name_or_path: str, *model_args, **kwargs):
        r"""    
        Instantiate a `SONARForConditionalGeneration` model from a pretrained encoder and decoder.
        """
        encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path)
        
        print("Number of encoder parameters:", sum(p.numel() for p in encoder.parameters()))
        
        model = SONARForText2Text.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.config.pretrained_encoder = encoder_pretrained_model_name_or_path
        model.config.tie_word_embeddings = False
        model.model.encoder = SONARSpeechEncoder.from_pretrained(encoder_pretrained_model_name_or_path, model, embed_tokens=model.model.shared)
        return model
    
    @classmethod
    def from_sonar_w2v_pretrained(cls, pretrained_model_name_or_path: str, w2v_pretrained_model_name_or_path: str, *model_args, **kwargs):
        
        model = M2M100ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        pooling_config = copy.deepcopy(model.config)
        pooling_config.decoder_layers = 3

        pooling_config.decoder_ffn_dim = 4096
        
        pooling_config.vocab_size = 1024

        # encoder = SONARSpeechEncoder.from_w2v_bert_pretrained(w2v_pretrained_model_name_or_path, pooling_config)
        
        w2v = AutoModel.from_pretrained(w2v_pretrained_model_name_or_path)  

        config = SONARSpeechConfig.from_encoder_decoder_configs(
            encoder_config=w2v.config,
            decoder_config=model.config,
            pooling_config=pooling_config
        )

        sonar_model = cls(config)
        sonar_model.model.encoder.encoder.load_state_dict(w2v.state_dict(), strict=True)
        sonar_model.model.decoder.load_state_dict(model.model.decoder.state_dict(), strict=True)
        sonar_model.model.shared.load_state_dict(model.model.shared.state_dict(), strict=False)
        sonar_model.lm_head.load_state_dict(model.lm_head.state_dict(), strict=False)
        sonar_model.generation_config = model.generation_config


        return sonar_model
    
    def freeze_decoder(self):
        # for param in self.model.decoder.parameters():
        #     param.requires_grad = False
        
        # self.model.decoder.eval()
        
        # self.model.shared.requires_grad = False
        
        # self.model.decoder.gradient_checkpointing = False
        
        for param in self.parameters():
            param.requires_grad = False

        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            M2M100 uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example Translation:

        ```python
        >>> from transformers import AutoTokenizer, M2M100ForConditionalGeneration

        >>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

        >>> text_to_translate = "Life is like a box of chocolates"
        >>> model_inputs = tokenizer(text_to_translate, return_tensors="pt")

        >>> # translate to French
        >>> gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
        >>> print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        if target_embeddings is not None:
            
            encoder_outputs = self.model.encoder(
                input_ids=input_features,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            


            mse_loss = self.mse_loss(encoder_outputs[0].squeeze(), target_embeddings)

            if not return_dict:
                return (mse_loss,) + encoder_outputs
            return Seq2SeqLMOutput(
            loss=mse_loss,
            logits=None,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=encoder_outputs,
            encoder_attentions=None,
        )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.lm_head(outputs[0])
        
        masked_lm_loss = None
        if labels is not None:

            labels = labels.to(lm_logits.device)

            masked_lm_loss = self.cross_entropy_loss(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            # print(f"Cross Entropy Loss: {masked_lm_loss if masked_lm_loss is not None else 'N/A'}")

            masked_lm_loss = masked_lm_loss.mean()

            
        # print(f"Masked LM Loss: {masked_lm_loss.item() if masked_lm_loss is not None else 'N/A'}")
        # print(f"MSE Loss: {mse_loss.item() if mse_loss is not None else 'N/A'}")

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _init_weights(self, module):
        pass
        
    # def generate(self, inputs = None, generation_config = None, logits_processor = None, stopping_criteria = None, prefix_allowed_tokens_fn = None, synced_gpus = None, assistant_model = None, streamer = None, negative_prompt_ids = None, negative_prompt_attention_mask = None, use_model_defaults = None, custom_generate = None, **kwargs):
    #     if "input_features" in inputs:
    #         inputs["input_ids"] = inputs.pop("input_features")
    #     return super().generate(inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, use_model_defaults, custom_generate, **kwargs)



