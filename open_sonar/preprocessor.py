
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from sonar.tokenizer import NllbTokenizerFast


import random


import random


class NllbPreprocessor:
    def __init__(self, tokenizer, data_args, padding="max_length", mode: str = "train", reverse=True, sort_buffer_size=1000):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.padding = padding
        self.mode = mode
        self.reverse = reverse
        self.buffer_size = sort_buffer_size

    def preprocess_batch(self, examples):
        """Preprocess batch for training mode."""
        source = self.tokenizer(
            text=examples['source.text'],
            langs=examples['source.nllb_code'],
            max_length=self.data_args.max_source_length,
            padding=self.padding,
            truncation=True,
        )
        
        target = self.tokenizer(
            text=examples['target.text'],
            langs=examples['target.nllb_code'],
            max_length=self.data_args.max_target_length,
            padding=self.padding,
            truncation=True,
        )

        return {
            "input_ids": source['input_ids'],
            "labels": target['input_ids'],
        }

    def preprocess_batch_predict(self, examples):
        """Preprocess batch for evaluation/prediction mode."""
        inputs = self.tokenizer(
            text=examples['source.text'],
            langs=examples['source.nllb_code'],
            max_length=self.data_args.max_source_length,
            padding=self.padding,
            truncation=True,
        )

        outputs = self.tokenizer(
            text=examples['target.text'],
            langs=examples['target.nllb_code'],
            max_length=self.data_args.max_target_length,
            padding=self.padding,
            truncation=True,
        )

        return {
            "input_ids": inputs['input_ids'],
            "labels": outputs['input_ids'],
            "labels_len": [len(label) for label in outputs['input_ids']]
        }
        
    def preprocess_batch_xsim(self, examples):
        """Preprocess batch for evaluation/prediction mode."""
        inputs = self.tokenizer(
            text=examples['sentence'],
            langs=examples['nllb_code'],
            max_length=self.data_args.max_source_length,
            padding=self.padding,
            truncation=True,
        )

        return {
            "input_ids": inputs['input_ids'],
            "labels": examples['id'],
            "labels_len": [len(label) for label in inputs['input_ids']]
        }


    def sort(self, examples, buffer_size=1000):
        """
        Sort examples by label length within buffer chunks to improve training efficiency
        while maintaining some randomness.
        
        Args:
            examples: Dictionary of examples with 'labels' key
            buffer_size: Size of buffer for sorting operation
        
        Returns:
            Sorted examples dictionary
        """
        
        # Calculate lengths of labels
        lengths = [(len(input_id)+len(target_id) ) / 2 for input_id, target_id in zip(examples["input_ids"], examples["labels"])]

        # Create indices list
        indices = list(range(len(lengths)))
        
        # Sort indices in chunks based on buffer size
        sorted_indices = []
        for i in range(0, len(indices), buffer_size):
            chunk_indices = indices[i:i + buffer_size]
            # Sort chunk by corresponding label length
            chunk_indices.sort(key=lambda idx: lengths[idx], reverse=self.reverse)
            sorted_indices.extend(chunk_indices)
        
        # Reorder all fields according to sorted indices
        sorted_examples = {}
        for key, values in examples.items():
            sorted_examples[key] = [values[i] for i in sorted_indices]
        
        return sorted_examples


    def save(self):
        """Save preprocessor state - implement as needed."""
        pass

    def __call__(self, examples, *args, **kwargs):
        """Main preprocessing entry point."""
        if self.mode == "train":
            # Apply alternation if enabled
            
            examples = self.preprocess_batch(examples, *args, **kwargs)
            
            if self.buffer_size > 0:
                examples = self.sort(examples, buffer_size=self.buffer_size)

            
            return examples
            
        elif self.mode in "eval":
            examples = self.preprocess_batch_predict(examples, *args, **kwargs)
            lengths = [len(target_id) for target_id in examples["labels"]]
            examples["labels_len"] = lengths

            examples["target_ids"] = examples["labels"].copy()

            return examples
        elif self.mode == "xsim":
            examples = self.preprocess_batch_xsim(examples, *args, **kwargs)
            lengths = [len(target_id) for target_id in examples["input_ids"]]
            examples["lengths"] = lengths

            examples["target_ids"] = examples["labels"].copy()

            return examples

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Expected 'train', 'eval', or 'predict'.")


    
import math
import random
from typing import List, Dict, Any, Optional, Callable, Union


class DAEProcessor:
    """
    A class to transform datasets by adding various types of noise to input_ids.
    All operations are performed using Python lists instead of PyTorch tensors.
    """
    
    def __init__(
        self,
        tokenizer,
        key_input: str = "input_ids",
        mask_ratio: float = 0.3,
        random_ratio: float = 0.1,
        insert_ratio: float = 0.0,
        rotate_ratio: float = 0.0,
        permute_sentence_ratio: float = 0.1,
        replace_length: int = 1,
        mask_length: str = "subword",
        poisson_lambda: float = 3.0,
        full_stop_token_id: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the DatasetNoiser with a Hugging Face tokenizer.
        
        Args:
            tokenizer: Hugging Face tokenizer instance
            mask_ratio: Ratio of tokens to mask
            random_ratio: Ratio of masked tokens to replace with random tokens
            insert_ratio: Ratio of tokens to insert
            rotate_ratio: Probability of applying rotation
            permute_sentence_ratio: Ratio for sentence permutation
            replace_length: 0=delete, 1=replace with mask, -1=keep original length
            mask_length: Type of masking ("subword", "word", "span-poisson")
            poisson_lambda: Lambda parameter for Poisson distribution
            full_stop_token_id: Token ID representing sentence boundaries
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        
        self.key_input = key_input
        
        # Extract token IDs from tokenizer
        self.vocab_size = len(tokenizer.get_vocab())
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
        # Handle different tokenizer types for BOS/EOS tokens
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        
        # Fallback for tokenizers without explicit BOS/EOS
        if self.bos_token_id is None:
            self.bos_token_id = getattr(tokenizer, 'cls_token_id', 0)
        if self.eos_token_id is None:
            self.eos_token_id = getattr(tokenizer, 'sep_token_id', self.bos_token_id)
        
        # Handle potential None values
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a mask token")
        if self.pad_token_id is None:
            self.pad_token_id = 0
        if self.bos_token_id is None:
            self.bos_token_id = 0
        if self.eos_token_id is None:
            self.eos_token_id = self.bos_token_id
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.insert_ratio = insert_ratio
        self.rotate_ratio = rotate_ratio
        self.permute_sentence_ratio = permute_sentence_ratio
        self.replace_length = replace_length
        self.mask_length = mask_length
        self.poisson_lambda = poisson_lambda
        
        # Set full stop token ID - try common punctuation tokens
        if full_stop_token_id is not None:
            self.full_stop_token_id = full_stop_token_id
        else:
            # Try to find period token
            period_candidates = ['.', '.</s>', '.', '▁.', '!', '!', '▁!', '▁. ', '▁! ', '?', '▁? ']
            self.full_stop_token_id = self.eos_token_id  # fallback
            for candidate in period_candidates:
                try:
                    token_id = tokenizer.convert_tokens_to_ids(candidate)
                    if token_id != tokenizer.unk_token_id:
                        self.full_stop_token_id = token_id
                        break
                except:
                    continue
        
        if seed is not None:
            random.seed(seed)
        
        # Validate parameters
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask_length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError("if using subwords, use replace_length=1 or 0")
        
        # Prepare Poisson distribution for span masking
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            self.mask_span_distribution = self._create_poisson_distribution(poisson_lambda)
    
    def _create_poisson_distribution(self, lambda_val: float) -> List[float]:
        """Create a Poisson distribution probability list."""
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-lambda_val)
        k_factorial = 1
        ps = []
        
        for k in range(0, 128):
            prob = e_to_the_minus_lambda * lambda_to_the_k / k_factorial
            ps.append(prob)
            lambda_to_the_k *= lambda_val
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        
        return ps
    
    def _sample_poisson(self) -> int:
        """Sample from the Poisson distribution."""
        if self.mask_span_distribution is None:
            return 1
        
        r = random.random()
        cumsum = 0
        for i, prob in enumerate(self.mask_span_distribution):
            cumsum += prob
            if r <= cumsum:
                return i
        return len(self.mask_span_distribution) - 1
    
    def noise(self, input_ids: List[int]) -> Dict[str, List[int]]:
        
        source = input_ids.copy()
        
                # Apply transformations in order
        if self.permute_sentence_ratio > 0.0:
            source = self.permute_sentences(source, self.permute_sentence_ratio)
        
        if self.mask_ratio > 0:
            source = self.add_whole_word_mask(source, self.mask_ratio)
        
        if self.insert_ratio > 0:
            source = self.add_insertion_noise(source, self.insert_ratio)
        
        if self.rotate_ratio > 0.0 and random.random() < self.rotate_ratio:
            source = self.add_rolling_noise(source)
            
        return source

    def __call__(self, examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        Apply noise transformations to input_ids.
        
        Args:
            input_ids: List of token IDs
            
        Returns:
            Dictionary containing 'source' (noised) and 'target' (original) sequences
        """
        if self.key_input not in examples:
            raise ValueError(f"Key '{self.key_input}' not found in examples")
        
        input_ids = examples[self.key_input].copy()  # Copy to avoid modifying original
        if not isinstance(input_ids, list):
            raise ValueError(f"Expected input_ids to be a list, got {type(input_ids)}")
        if len(input_ids) == 0:
            raise ValueError("input_ids cannot be empty")
        
        
        if isinstance(input_ids[0], list):
            # Handle batch of sequences
            noised_input_ids = [self.noise(seq) for seq in input_ids]
            noised_input_masks = [[1] * len(seq) for seq in noised_input_ids]
        else:
            noised_input_ids = self.noise(input_ids)
            noised_input_masks = [1] * len(noised_input_ids)  # Single sequence mask

        examples[f"noised_{self.key_input}"] = noised_input_ids
        
        return examples

    def permute_sentences(self, tokens: List[int], p: float = 1.0) -> List[int]:
        """Permute sentences in the token sequence."""
        # Find sentence boundaries (full stops)
        full_stops = [i for i, token in enumerate(tokens) if token == self.full_stop_token_id]
        
        if len(full_stops) <= 1:
            return tokens
        
        # Ensure the sequence ends with a full stop for proper sentence detection
        if tokens[-2] != self.full_stop_token_id:
            full_stops.append(len(tokens) - 2)
        
        # Find sentence end positions
        sentence_ends = []
        for i in range(1, len(tokens)):
            if tokens[i] == self.full_stop_token_id and tokens[i-1] != self.full_stop_token_id:
                sentence_ends.append(i + 1)
        
        if len(sentence_ends) <= 1:
            return tokens
        
        num_sentences = len(sentence_ends)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        
        # Create permutation
        indices = list(range(num_sentences))
        substitutions = random.sample(range(num_sentences), min(num_to_permute, num_sentences))
        permuted_substitutions = substitutions.copy()
        random.shuffle(permuted_substitutions)
        
        for i, j in zip(substitutions, permuted_substitutions):
            indices[i] = j
        
        # Reconstruct sequence with permuted sentences
        result = [tokens[0]]  # Keep BOS token
        start_idx = 1
        
        for i in indices:
            sentence_start = sentence_ends[i-1] if i > 0 else 1
            sentence_end = sentence_ends[i]
            sentence = tokens[sentence_start:sentence_end]
            result.extend(sentence)
        
        return result
    
    def add_whole_word_mask(self, tokens: List[int], p: float) -> List[int]:
        """Add whole word masking to the sequence."""
        # For simplicity, treat each token as a word start
        # In practice, you might want to use word boundary information
        word_starts = [i for i in range(1, len(tokens) - 1)]  # Exclude BOS and EOS
        
        if not word_starts:
            return tokens
        
        num_to_mask = int(math.ceil(len(word_starts) * p))
        if num_to_mask == 0:
            return tokens
        
        # Sample lengths for span masking
        if self.mask_span_distribution is not None:
            lengths = [self._sample_poisson() for _ in range(num_to_mask)]
            
            # Adjust lengths to fit masking budget
            total_length = sum(lengths)
            while total_length < num_to_mask:
                lengths.extend([self._sample_poisson() for _ in range(num_to_mask)])
                total_length = sum(lengths)
            
            # Trim to budget
            cumsum = 0
            for i, length in enumerate(lengths):
                cumsum += length
                if cumsum >= num_to_mask:
                    lengths[i] = num_to_mask - (cumsum - length)
                    lengths = lengths[:i+1]
                    break
            
            lengths = [l for l in lengths if l > 0]
            num_to_mask = len(lengths)
        else:
            lengths = [1] * num_to_mask
        
        # Select positions to mask
        selected_positions = random.sample(word_starts, min(num_to_mask, len(word_starts)))
        
        # Apply masking
        result = tokens.copy()
        positions_to_remove = set()
        
        for pos, length in zip(selected_positions, lengths):
            for i in range(length):
                if pos + i < len(result) - 1:  # Don't mask EOS
                    if random.random() < self.random_ratio:
                        # Replace with random token
                        result[pos + i] = random.randint(1, self.vocab_size - 1)
                    else:
                        # Replace with mask token
                        result[pos + i] = self.mask_token_id
                    
                    if self.replace_length == 0 and i > 0:
                        positions_to_remove.add(pos + i)
        
        # Remove positions if replace_length == 0
        if self.replace_length == 0:
            result = [token for i, token in enumerate(result) if i not in positions_to_remove]
        
        return result
    
    def add_insertion_noise(self, tokens: List[int], p: float) -> List[int]:
        """Add insertion noise to the sequence."""
        if p == 0.0:
            return tokens
        
        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))
        
        if n == 0:
            return tokens
        
        # Choose positions to insert (excluding BOS and EOS positions)
        possible_positions = list(range(1, num_tokens))
        insert_positions = random.sample(possible_positions, min(n, len(possible_positions)))
        insert_positions.sort(reverse=True)  # Insert from right to left to maintain indices
        
        result = tokens.copy()
        num_random = int(math.ceil(n * self.random_ratio))
        
        for i, pos in enumerate(insert_positions):
            if i < num_random:
                # Insert random token
                new_token = random.randint(1, self.vocab_size - 1)
            else:
                # Insert mask token
                new_token = self.mask_token_id
            
            result.insert(pos, new_token)
        
        return result
    
    def add_rolling_noise(self, tokens: List[int]) -> List[int]:
        """Add rolling/rotation noise to the sequence."""
        if len(tokens) <= 3:  # Need at least BOS + 1 token + EOS
            return tokens
        
        # Choose random offset (excluding BOS and EOS)
        max_offset = len(tokens) - 2
        offset = random.randint(1, max_offset)
        
        # Rotate the middle part (excluding BOS and EOS)
        middle = tokens[1:-1]
        rotated_middle = middle[offset:] + middle[:offset]
        
        return [tokens[0]] + rotated_middle + [tokens[-1]]


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Initialize with a Hugging Face tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    
    # noiser = DatasetNoiser(
    #     tokenizer=tokenizer,
    #     mask_ratio=0.15,
    #     random_ratio=0.1,
    #     insert_ratio=0.1,
    #     rotate_ratio=0.1,
    #     permute_sentence_ratio=1.0
    # )
    
    # Example input
    text = "Hello world! This is a test sentence. Bonjour le monde! C'est une phrase de test."
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    
    # Apply noise
    # result = noiser(input_ids)
    
    # print("Original text:", text)
    # print("Original IDs: ", input_ids)
    # print("Target:       ", result['target'])
    # print("Source:       ", result['source'])
    # print("Target text:  ", tokenizer.decode(result['target']))
    # print("Source text:  ", tokenizer.decode(result['source']))