from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import (
    M2M100Config,)

from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Encoder,
    M2M100ScaledWordEmbedding,
    M2M100ForConditionalGeneration,
    M2M100Model,
    shift_tokens_right,
    logger)

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput

from transformers.utils import auto_docstring

from open_sonar.loss import CrossEntropyLoss

class Pooling(nn.Module):
    """
    Pooling layer for sequence representations.
    
    Supports multiple pooling strategies: mean, cls, last, max, and none.
    """
    
    def __init__(self, pooling_type: str = "mean"):
        super().__init__()
        valid_types = {"mean", "cls", "last", "max", "none"}
        if pooling_type not in valid_types:
            raise ValueError(f"pooling_type must be one of {valid_types}, got {pooling_type}")
        self.pooling_type = pooling_type
    
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
        if self.pooling_type == "none":
            return hidden_states
        
        if self.pooling_type == "cls":
            return hidden_states[:, 0, :].unsqueeze(1)
        
        elif self.pooling_type == "last":
            return hidden_states[:, -1, :].unsqueeze(1)
        
        elif self.pooling_type in ["mean", "max"]:
            if attention_mask is None:
                raise ValueError(f"attention_mask is required for {self.pooling_type} pooling")
            
            # Expand attention mask to match hidden_states dimensions
            mask = attention_mask.unsqueeze(-1)
            
            if self.pooling_type == "mean":
                # Apply mask and compute mean over valid tokens
                masked_hidden = hidden_states * mask
                sum_hidden = masked_hidden.sum(dim=1, keepdim=True)
                sum_mask = mask.sum(dim=1, keepdim=True)
                
                return sum_hidden / sum_mask
            
            elif self.pooling_type == "max":
                # Apply mask (set masked positions to large negative value)
                masked_hidden = hidden_states.masked_fill(mask == 0, float('-inf'))
                return masked_hidden.max(dim=1, keepdim=True)[0]

class SONARTextEncoder(M2M100Encoder):
    """
    Transformer encoder with pooling capabilities.
    
    Inherits from M2M100Encoder and adds configurable pooling functionality.
    
    Args:
        config: M2M100Config with optional pooling_type attribute
        embed_tokens: Optional embedding layer
    """

    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        
        # Initialize pooling layer
        pooling_type = getattr(config, 'pooling_type', 'mean')
        self.pooling = Pooling(pooling_type)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass with optional pooling.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask for padding tokens
            head_mask: Mask for attention heads
            inputs_embeds: Pre-computed embeddings
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return ModelOutput object
            pool: Pooling strategy override (if None, uses config default)
        
        Returns:
            Model output with pooled representations
        """
        # Get encoder output
        encoder_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Extract hidden states
        if return_dict:
            hidden_states = encoder_output.last_hidden_state

        else:
            hidden_states = encoder_output[0]
        
        pooled_output = self.pooling(hidden_states, attention_mask)

        if return_dict:
            encoder_output.last_hidden_state = pooled_output
        else:
            encoder_output = (pooled_output,) + encoder_output[1:]

        return encoder_output

class SONARModel(M2M100Model):
    """SONAR model based on M2M100."""

    def __init__(self, config: M2M100Config):
        super().__init__(config)

        self.encoder = SONARTextEncoder(config, self.shared)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
        return_logits: Optional[bool] = False,
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
        
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        if return_logits:
            lm_logits = self.decoder.lm_head(outputs[0])
            if not return_dict:
                outputs = (lm_logits,) + outputs[1:]
            else:
                outputs.last_hidden_state = lm_logits
                
        return outputs
        
        


    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
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
            if (encoder_outputs[0].size(1) != 1 and encoder_outputs[0].dim() == 3):
                logger.warning_once(
                    f"Encoder is not pooled"
                )
                encoder_attention_mask = attention_mask
            else:
                encoder_attention_mask = attention_mask[:, :1]
        
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
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

class SONARForText2Text(M2M100ForConditionalGeneration):
    """SONAR model for conditional generation tasks."""
    
    # _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: M2M100Config):
        super().__init__(config)
        
        self.model = SONARModel(config)

        self.cross_entropy_loss = CrossEntropyLoss(
            label_smoothing=0.1,
            ignore_index=-100
        )
        
        self.mse_loss = nn.MSELoss()
        

        self.mse_ratio = getattr(config, 'mse_ratio', 0.2)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.LongTensor] = None, # IDs of target inputs only for MSE loss computation
        target_attention_mask: Optional[torch.Tensor] = None, # Attention mask of target inputs only for MSE loss computation
        mse_mask: Optional[torch.Tensor] = None, # Mask to apply on the MSE loss computation for paired inputs (1 for if regular input, 0 for noised input)
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

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
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
        
        
        loss = None

        # COompute the cross-entropy loss if labels are provided, if 
        if labels is not None:
            labels = labels.to(lm_logits.device)
            mt_loss = self.cross_entropy_loss(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = mt_loss

        # Compute the MSE loss in the case of paired inputs (when mse_mask is provided)
        if mse_mask is not None and target_ids is None:
            mse_mask = mse_mask.view(-1, 1, 1).to(outputs.encoder_last_hidden_state.device)
            encoder_outputs = outputs.encoder_last_hidden_state.squeeze()
            batch_size = encoder_outputs.size(0)
    
            paired = encoder_outputs[:batch_size//2*2].view(batch_size//2, 2, *encoder_outputs.shape[1:]) * mse_mask

            mse_loss = self.mse_loss(paired[:, 0], paired[:, 1])
            
            loss += self.mse_ratio * mse_loss

        # Compute the MSE loss in the case of not paired inputs (target_ids should be provided)
        if target_ids is not None and labels is not None:

            target_ids = target_ids.to(lm_logits.device)
            target_encoder_outputs = self.model.encoder(
                input_ids=target_ids,
                attention_mask=target_attention_mask,
                return_dict=return_dict,
            )

            mse_loss = self.mse_loss(outputs.encoder_last_hidden_state, target_encoder_outputs.last_hidden_state)

            loss += self.mse_ratio * mse_loss


        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(self, target_lang_ids: Union[int, list, torch.Tensor] = 256047, **kwargs) -> torch.LongTensor:

        """
        Generate sequences of token ids for each example in the batch, given the model and the input ids.
        Args:
            target_lang_ids (int, list, or torch.Tensor): Language ID(s) for the target language.
                Can be a single integer (same language for all examples), a list of integers (one per example),
                or a tensor of shape (batch_size,). Defaults to 256047 (English).
            **kwargs: Additional keyword arguments passed to the underlying generate method.
        Returns:
            torch.LongTensor: Generated sequences of token ids.
        """
        
        if 'decoder_input_ids' in kwargs:
            raise ValueError("decoder_input_ids should not be provided for generation.")
        
        if type(target_lang_ids) is int:
            target_lang_ids = torch.tensor([target_lang_ids], device=self.device)
        elif type(target_lang_ids) is list:
            target_lang_ids = torch.tensor(target_lang_ids, device=self.device)
        elif type(target_lang_ids) is torch.Tensor:
            target_lang_ids = target_lang_ids.to(self.device)
            
        batch_size = kwargs['input_ids'].size(0) if 'input_ids' in kwargs else kwargs['encoder_outputs'].last_hidden_state.size(0)
        if target_lang_ids.size(0) == 1 and batch_size > 1:
            target_lang_ids = target_lang_ids.expand(batch_size)

        kwargs['decoder_input_ids'] = target_lang_ids.squeeze().unsqueeze(-1)

        return super().generate(**kwargs)

    def encode(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Encode the given input IDs into embeddings.
        """
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        return encoder_outputs.last_hidden_state.squeeze()

    def decode(self, encoder_outputs: Optional[torch.Tensor] = None, target_lang_ids: int = 256047, **kwargs):
        """
        Decode the given encoder outputs (embedings) into target language IDs.
        """

        if encoder_outputs is None:
            raise ValueError("encoder_outputs should be provided for decoding.")
        else:
            if len(encoder_outputs.size()) == 2:
                encoder_outputs = encoder_outputs.unsqueeze(1)

        decoder_outputs = self.generate(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            target_lang_ids=target_lang_ids,
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
        )
        
        return decoder_outputs


    @classmethod
    def from_m2m100_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        model = M2M100ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        generation_config = model.generation_config
        
        generation_config.early_stopping = True
        generation_config.num_beams = 5
        generation_config.max_length = 500
        
        config = model.config
        config.pooling_type = getattr(config, 'pooling_type', 'mean')
        
        sonar_model = cls(model.config)
        sonar_model.load_state_dict(model.state_dict(), strict=False)
        sonar_model.generation_config = generation_config
        return sonar_model
    
    


