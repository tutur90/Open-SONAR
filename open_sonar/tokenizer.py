import numpy as np
import math
import random
from transformers.models.nllb.tokenization_nllb import NllbTokenizer as _NllbTokenizer
from transformers.models.nllb.tokenization_nllb_fast import NllbTokenizerFast as _NllbTokenizerFast


# NllbTokenizer which encode for different languages in the same batch, work with both fast and regular tokenizers
class NllbTokenizer(_NllbTokenizer):
    """
    NllbTokenizer which encode for different languages in the same batch, work with both fast and regular tokenizers
    """

    def __init__(self, *args, src_lang="eng_Latn", tgt_lang="eng_Latn", **kwargs):
        super().__init__(*args, **kwargs)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def add_langs(self, langs, output, feature='input_ids'):
        
        if langs is None:
            langs = [self.src_lang if feature == 'input_ids' else self.tgt_lang] * len(output[feature])
            if isinstance(langs, str):
                langs = [langs] * len(output[feature])
            elif isinstance(langs, list) and len(langs) != len(output[feature]):
                raise ValueError("If langs is a list, it must have the same length as the batch size.")
            
        for i, lang in enumerate(langs):
            output[feature][i][0] = self.convert_tokens_to_ids(lang)
        return output
        
    def __call__(self, text, langs=None, langs_target=None, *args, **kwargs):
        """Encode text with optional source and target languages."""
        output = super().__call__(text, *args, **kwargs)

        output = self.add_langs(langs, output)

        output = self.add_langs(langs_target, output, feature='labels')

        return output
    

class NllbTokenizerFast(_NllbTokenizerFast):
    """
    Optimized version of the NllbTokenizer which encode for different languages in the same batch, work only with fast tokenizers
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_dynamic_encode(self, text, langs, max_length=None, function=lambda x: x):
        """Dynamically encode a batch of text with corresponding languages."""
        
        if len(text) != len(langs):
            raise ValueError("Text and language must have the same length.")
        
        encoded_lang = []
        encoded_text = []
        attention_mask = []
        
        if max_length is None:
            max_length = self.model_max_length

        for text_encoding, lang_encoding in zip(self._tokenizer.encode_batch(text, add_special_tokens=False), self._tokenizer.encode_batch(langs, add_special_tokens=False)):
            encoded_lang.append(lang_encoding.ids)
            encoded_text.append(lang_encoding.ids + function( text_encoding.ids if len(text_encoding.ids) >= max_length-2 else text_encoding.ids[:max_length-2] ) + [self.eos_token_id])
            attention_mask.append([1] * len(encoded_text[-1]))

        return {
            "input_ids": encoded_text,
            "attention_mask": attention_mask,
            "lang": encoded_lang
        }

    def __call__(self, text, langs, max_length=None, truncation=True, padding=True, function=lambda x: x):
        return self.batch_dynamic_encode(text, langs, max_length=max_length, function=function)