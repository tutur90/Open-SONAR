import numpy as np
import math
import random
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
from  transformers.models.nllb.tokenization_nllb_fast import NllbTokenizerFast as _NllbTokenizerFast



class NllbTokenizerFast(_NllbTokenizerFast):

    def __init__(self, *args, src_lang="eng_Latn", tgt_lang="eng_Latn", **kwargs):
        super().__init__(*args, **kwargs)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    def __call__(self, text, langs=None, langs_target=None, *args, **kwargs):
        
        # Check if self.src_lang attribute exists
        if self.src_lang is None:
            print("Warning: src_lang is not set. Defaulting to 'eng_Latn'.")
            self.src_lang = 'und'
        if self.tgt_lang is None:
            print("Warning: tgt_lang is not set. Defaulting to 'eng_Latn'.")
            self.tgt_lang = 'und'
            

        output = super().__call__(text, *args, **kwargs)
        
        
        if langs is not None:

            for i, lang in enumerate(langs):
                if lang is not None:
                    output['input_ids'][i][0] = self.convert_tokens_to_ids(lang)
                else:
                    raise ValueError(f"Language code {lang} not found in tokenizer's language codes.")
                
        if langs_target is not None:
            for i, lang in enumerate(langs_target):
                if lang is not None:
                    output['labels'][i][0] = self.convert_tokens_to_ids(lang)
                else:
                    raise ValueError(f"Target language code {lang} not found in tokenizer's language codes.")
        return output