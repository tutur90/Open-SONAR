import numpy as np
import math
import random
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
from  transformers.models.nllb.tokenization_nllb_fast import NllbTokenizerFast as NllbTF



class NllbTokenizerFast(NllbTF):
    """
    A fast tokenizer for NLLB models with additional noise injection capabilities.
    Inherits from NllbTokenizerFast and extends it with methods for dynamic encoding
    and noise injection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_dynamic_encode(self, text, langs, max_length=None, function=lambda x: x):

        """
        Encode text with dynamic noise injection capabilities.
        
        Args:
            text (str): Input text to encode
            add_special_tokens (bool): Whether to add special tokens
            **kwargs: Additional arguments for noise control
                - mask_ratio: Probability of masking tokens
                - random_ratio: Probability of random token replacement
                - insert_ratio: Probability of inserting noise tokens
                - rotate_ratio: Probability of applying rotation
                - permute_sentence_ratio: Probability of permuting sentences
        
        Returns:
            dict: Dictionary with 'input_ids', 'attention_mask', and 'labels'
        """
        
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


class DenoisingTokenizer(NllbTokenizerFast):
    def __init__(self, *args, **kwargs):
        # Extract noise parameters from kwargs
        self.mask_ratio = kwargs.pop('mask_ratio', 0.3)
        self.random_ratio = kwargs.pop('random_ratio', 0.0) 
        self.insert_ratio = kwargs.pop('insert_ratio', 0.05)
        self.rotate_ratio = kwargs.pop('rotate_ratio', 0.0)
        self.permute_sentence_ratio = kwargs.pop('permute_sentence_ratio', 0.1)
        self.replace_length = kwargs.pop('replace_length', 1)
        self.mask_length = kwargs.pop('mask_length', 'subword')
        self.poisson_lambda = kwargs.pop('poisson_lambda', 3.0)
        self.bpe_type = kwargs.pop('bpe', 'sentencepiece')
        
        super().__init__(*args, **kwargs)
        
        # Initialize mask span distribution for span-poisson
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            self._init_poisson_distribution()
            
        # Set full stop index based on BPE type
        if self.bpe_type != "gpt2":
            self.full_stop_index = self.eos_token_id
        else:
            # For GPT-2 BPE, find the period token
            self.full_stop_index = self.convert_tokens_to_ids(".")
        
        # Validate parameters
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask_length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace_length=1 or 0")
    
    def _init_poisson_distribution(self):
        """Initialize Poisson distribution for span masking."""
        _lambda = self.poisson_lambda
        
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
                
        self.mask_span_probs = np.array(ps)
        self.mask_span_probs /= self.mask_span_probs.sum()  # Normalize
    
    def tokenize_noise(self, text, noise_level=0.1):
        """
        Tokenizes the input text with a specified noise level.
        """
        tokens = self.tokenize(text)
        num_noise_tokens = int(len(tokens) * noise_level)
       
        if num_noise_tokens > 0:
            noise_indices = np.random.choice(len(tokens), num_noise_tokens, replace=False)
            for idx in noise_indices:
                tokens[idx] = self.unk_token
                
        return tokens
    
    def add_noise(self, tokens, **kwargs):
        """
        Adds noise to the encoded text based on the specified noise level.
        
        Args:
            encoded_text (list): List of token IDs representing the encoded text.
            noise_level (float): Proportion of tokens to be replaced with noise.
        
        Returns:
            list: Encoded text with noise added.
        """
        
                # Override instance parameters with kwargs
        mask_ratio = kwargs.get('mask_ratio', self.mask_ratio)
        random_ratio = kwargs.get('random_ratio', self.random_ratio)
        insert_ratio = kwargs.get('insert_ratio', self.insert_ratio)
        rotate_ratio = kwargs.get('rotate_ratio', self.rotate_ratio)
        permute_sentence_ratio = kwargs.get('permute_sentence_ratio', self.permute_sentence_ratio)
        
        
        # Apply noise transformations
        if permute_sentence_ratio > 0.0:
            tokens = self._permute_sentences(tokens, permute_sentence_ratio)
            
        if mask_ratio > 0:
            tokens = self._add_whole_word_mask(tokens, mask_ratio, random_ratio)
            
        if insert_ratio > 0:
            tokens = self._add_insertion_noise(tokens, insert_ratio, random_ratio)
            
        if rotate_ratio > 0.0 and np.random.random() < rotate_ratio:
            tokens = self._add_rolling_noise(tokens)

        return tokens
        
    

    def encode_with_noise(self, text, lang=None, add_special_tokens=True, **kwargs):
        """
        Encode text with comprehensive noise injection capabilities.
        
        Args:
            text (str): Input text to encode
            add_special_tokens (bool): Whether to add special tokens
            **kwargs: Additional arguments for noise control
                - mask_ratio: Probability of masking tokens
                - random_ratio: Probability of random token replacement
                - insert_ratio: Probability of inserting noise tokens
                - rotate_ratio: Probability of applying rotation
                - permute_sentence_ratio: Probability of permuting sentences
        
        Returns:
            dict: Dictionary with 'input_ids', 'attention_mask', and 'labels'
        """
        # Initial encoding
        tokens = self.batch_dynamic_encode(text, lang= lang if lang is not None else [self.src_lang] * len(text))

        return tokens

    def _permute_sentences(self, source, p=1.0):
        """Permute sentences within the source (no BOS/EOS, no numpy)."""
        # Find sentence boundaries by looking for sentence-ending punctuation
        sentence_enders = []
        for token_id in [self.full_stop_index]:
            if token_id is not None:
                sentence_enders.append(token_id)
        try:
            period_id = self.convert_tokens_to_ids(".")
            exclaim_id = self.convert_tokens_to_ids("!")
            question_id = self.convert_tokens_to_ids("?")
            for tid in [period_id, exclaim_id, question_id]:
                if tid is not None and tid != self.unk_token_id:
                    sentence_enders.append(tid)
        except Exception:
            pass
        sentence_enders = list(set(sentence_enders))

        if not sentence_enders:
            return source[:]

        # Find positions where sentences end
        sentence_ends = []
        for i in range(len(source)):
            if source[i] in sentence_enders:
                sentence_ends.append(i + 1)  # Include the punctuation in the sentence

        if len(sentence_ends) == 0:
            return source[:]

        if sentence_ends[-1] != len(source):
            sentence_ends.append(len(source))

        # Extract sentences
        sentences = []
        start_idx = 0
        for end_idx in sentence_ends:
            if end_idx > start_idx:
                sentences.append(source[start_idx:end_idx])
                start_idx = end_idx

        if len(sentences) <= 1:
            return source[:]

        # Permute sentences
        num_sentences = len(sentences)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        indices = list(range(num_sentences))
        substitutions = random.sample(indices, min(num_to_permute, num_sentences))
        permuted = substitutions[:]
        random.shuffle(permuted)
        ordering = indices[:]
        for i, idx in enumerate(substitutions):
            ordering[idx] = permuted[i]

        # Reconstruct with permuted sentences
        result = []
        for i in ordering:
            result.extend(sentences[i])
        return result

    def _word_starts(self, source):
        """Identify word start positions (no BOS/EOS, no numpy)."""
        # Each token is a word start
        return [True] * len(source)

    def _add_whole_word_mask(self, source, p, random_ratio):
        """Add whole word masking to source (no numpy, no BOS/EOS)."""
        is_word_start = self._word_starts(source)
        num_to_mask = int(math.ceil(sum(is_word_start) * p))
        num_inserts = 0

        if num_to_mask == 0:
            return source[:]

        if self.mask_span_distribution is not None:
            # Span-poisson masking
            lengths = []
            while sum(lengths) < num_to_mask:
                for _ in range(num_to_mask):
                    idx = random.choices(range(len(self.mask_span_probs)), weights=self.mask_span_probs)[0]
                    lengths.append(idx + 1)
            # Trim to masking budget
            cum_length = 0
            valid_lengths = []
            for l in lengths:
                if cum_length + l <= num_to_mask:
                    valid_lengths.append(l)
                    cum_length += l
                else:
                    break
            if cum_length < num_to_mask:
                valid_lengths.append(num_to_mask - cum_length)
            lengths = [l for l in valid_lengths if l > 0]
            num_inserts = num_to_mask - len(lengths)
            num_to_mask = len(lengths)
            if num_to_mask == 0:
                return self._add_insertion_noise(source, num_inserts / len(source), random_ratio)
        else:
            lengths = [1] * num_to_mask

        # Select random word starts to mask
        word_starts = [i for i, b in enumerate(is_word_start) if b]
        if not word_starts:
            return source[:]
        indices = random.sample(word_starts, min(num_to_mask, len(word_starts)))
        mask_random = [random.random() < random_ratio for _ in indices]

        source_out = source[:]
        to_keep = [True] * len(source_out)

        if self.replace_length == 0:
            for idx in indices:
                to_keep[idx] = False
        else:
            for i, idx in enumerate(indices):
                if mask_random[i]:
                    source_out[idx] = random.randint(1, self.vocab_size - 1)
                else:
                    source_out[idx] = self.mask_token_id

        # Handle span masking
        if self.mask_span_distribution is not None:
            current_indices = indices[:]
            current_mask_random = mask_random[:]
            current_lengths = [l - 1 for l in lengths]
            while current_indices:
                # Move to next position
                current_indices = [i + 1 for i in current_indices]
                # Check bounds
                valid = [i for i in range(len(current_indices)) if current_indices[i] < len(source_out)]
                current_indices = [current_indices[i] for i in valid]
                current_mask_random = [current_mask_random[i] for i in valid]
                current_lengths = [current_lengths[i] for i in valid]
                if not current_indices:
                    break
                # Update lengths based on word boundaries
                for i in range(len(current_indices)):
                    if is_word_start[current_indices[i]]:
                        current_lengths[i] -= 1
                # Keep only uncompleted spans
                uncompleted = [i for i in range(len(current_lengths)) if current_lengths[i] >= 0]
                current_indices = [current_indices[i] for i in uncompleted]
                current_mask_random = [current_mask_random[i] for i in uncompleted]
                current_lengths = [current_lengths[i] for i in uncompleted]
                if not current_indices:
                    break
                if self.replace_length != -1:
                    for idx in current_indices:
                        to_keep[idx] = False
                else:
                    for i, idx in enumerate(current_indices):
                        if current_mask_random[i]:
                            source_out[idx] = random.randint(1, self.vocab_size - 1)
                        else:
                            source_out[idx] = self.mask_token_id
        else:
            # Simple word-level masking
            current_indices = indices[:]
            current_mask_random = mask_random[:]
            while current_indices:
                current_indices = [i + 1 for i in current_indices]
                # Check bounds and word boundaries
                valid = [i for i in range(len(current_indices)) if current_indices[i] < len(source_out) and not is_word_start[current_indices[i]]]
                current_indices = [current_indices[i] for i in valid]
                current_mask_random = [current_mask_random[i] for i in valid]
                if not current_indices:
                    break
                if self.replace_length != -1:
                    for idx in current_indices:
                        to_keep[idx] = False
                else:
                    for i, idx in enumerate(current_indices):
                        if current_mask_random[i]:
                            source_out[idx] = random.randint(1, self.vocab_size - 1)
                        else:
                            source_out[idx] = self.mask_token_id

        # Remove masked tokens if needed
        source_final = [tok for i, tok in enumerate(source_out) if to_keep[i]]

        if num_inserts > 0:
            source_final = self._add_insertion_noise(source_final, num_inserts / len(source_final), random_ratio)

        return source_final

    def _add_rolling_noise(self, tokens):
        """Add rolling/rotation noise to tokens (no BOS/EOS, no numpy)."""
        if len(tokens) <= 1:
            return tokens[:]
        offset = random.randint(1, len(tokens) - 1)
        return tokens[offset:] + tokens[:offset]

    def _add_insertion_noise(self, tokens, p, random_ratio):
        """Add insertion noise to tokens (no numpy, no BOS/EOS)."""
        if p == 0.0:
            return tokens[:]
        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))
        if n == 0:
            return tokens[:]
        # Generate random positions for insertions
        positions = sorted(random.sample(range(num_tokens + n), n))
        result = []
        token_idx = 0
        insert_idx = 0
        for i in range(num_tokens + n):
            if insert_idx < n and i == positions[insert_idx]:
                if random.random() < random_ratio:
                    result.append(random.randint(1, self.vocab_size - 1))
                else:
                    result.append(self.mask_token_id)
                insert_idx += 1
            else:
                result.append(tokens[token_idx])
                token_idx += 1
        return result

    def _add_permuted_noise(self, tokens, p):
        """Add permutation noise to tokens (no numpy, no BOS/EOS)."""
        num_words = len(tokens)
        if num_words <= 1:
            return tokens[:]
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        indices = list(range(num_words))
        substitutions = random.sample(indices, min(num_to_permute, num_words))
        permuted = substitutions[:]
        random.shuffle(permuted)
        tokens_out = tokens[:]
        for i, idx in enumerate(substitutions):
            tokens_out[idx] = tokens[permuted[i]]
        return tokens_out


if __name__ == "__main__":
    # Initialize tokenizer with noise parameters
    tokenizer = DenoisingTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        mask_ratio=0.30,
        random_ratio=0.0,
        insert_ratio=0.0,
        rotate_ratio=0.0,
        permute_sentence_ratio=1.0,
        mask_length="span-poisson",
        poisson_lambda=3.0
    )
    
    text = ["Hello world! This is a test sentence. Another sentence here.",
            "This is another example sentence to test the tokenizer's noise capabilities.",
            "Let's see how it handles different types of noise."]
    
    lang = ["eng_Latn", "eng_Latn", "eng_Latn"]

    unnoized = tokenizer.batch_dynamic_encode(text, langs=lang)
    
    print("Unnoized tokens:", unnoized['input_ids'])

    
    result = tokenizer.encode_with_noise(
        text,
        langs=lang,
    )
    

    # Comprehensive noise encoding
    result = tokenizer.encode_with_noise(
        text
    )
    
    print("Original text:", text)
    print("Unnoised tokens:", unnoized['input_ids'])
    print("Noised tokens:", result['input_ids'])
    print("Source tokens:", tokenizer.batch_decode(result['input_ids']))
    # print("Target tokens:", tokenizer.batch_decode(result['labels']))