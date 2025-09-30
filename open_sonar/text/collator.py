from math import exp, trunc
import random
import re
import torch
from typing import List, Dict, Any, Union, Optional
import math

class DAEProcessor:
    """
    A class to transform datasets by adding various types of noise to input_ids.
    All operations are performed using Python lists instead of PyTorch tensors.
    """
    
    def __init__(
        self,
        tokenizer,
        key_input: str = "input_ids",
        dae_ratio: float = 0.01,
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
        Initialize the DAEProcessor with a Hugging Face tokenizer.
        
        Args:
            tokenizer: Hugging Face tokenizer instance
            key_input: Key for input data in the dataset
            dae_ratio: Ratio for denoising autoencoder application
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
        self.dae_ratio = dae_ratio
        
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
            period_candidates = ['.', '.</s>', '▁.', '!', '▁!', '?', '▁?']
            self.full_stop_token_id = self.eos_token_id  # fallback
            for candidate in period_candidates:
                try:
                    token_id = tokenizer.convert_tokens_to_ids(candidate)
                    if token_id != tokenizer.unk_token_id and token_id is not None:
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
    
    def noise(self, input_ids: List[int]) -> List[int]:
        """Apply noise to input_ids and return the noised version."""
        if not input_ids:
            return input_ids
            
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

    def __call__(self, example: Dict[str, List[int]]) -> List[int]:
        """Apply noise to examples and return noised input_ids."""
        return self.noise(example)

    def permute_sentences(self, tokens: List[int], p: float = 1.0) -> List[int]:
        """Permute sentences in the token sequence."""
        if len(tokens) <= 2:  # Need more than just BOS/EOS
            return tokens
            
        # Find sentence boundaries (full stops)
        sentence_ends = []
        for i in range(1, len(tokens) - 1):  # Exclude BOS and EOS
            if tokens[i] == self.full_stop_token_id:
                sentence_ends.append(i + 1)
        
        if len(sentence_ends) <= 1:
            return tokens
        
        # Add final position if no sentence end found
        if not sentence_ends or sentence_ends[-1] < len(tokens) - 1:
            sentence_ends.append(len(tokens) - 1)
        
        num_sentences = len(sentence_ends)
        if num_sentences <= 1:
            return tokens
            
        num_to_permute = min(math.ceil(num_sentences * p), num_sentences)
        
        # Create permutation
        indices = list(range(num_sentences))
        substitutions = random.sample(range(num_sentences), num_to_permute)
        permuted_substitutions = substitutions.copy()
        random.shuffle(permuted_substitutions)
        
        for i, j in zip(substitutions, permuted_substitutions):
            indices[i] = j
        
        # Reconstruct sequence with permuted sentences
        result = [tokens[0]]  # Keep BOS token
        start_idx = 1
        
        for i in indices:
            sentence_end = sentence_ends[i]
            sentence = tokens[start_idx:sentence_end]
            result.extend(sentence)
            start_idx = sentence_end
        
        # Add EOS token if present
        if tokens[-1] != tokens[0]:  # If EOS is different from BOS
            result.append(tokens[-1])
        
        return result
    
    def add_whole_word_mask(self, tokens: List[int], p: float) -> List[int]:
        """Add whole word masking to the sequence."""
        if len(tokens) <= 2:  # Need more than just BOS/EOS
            return tokens
            
        # For simplicity, treat each token as a word start (excluding BOS and EOS)
        word_starts = list(range(1, len(tokens) - 1))
        
        if not word_starts:
            return tokens
        
        num_to_mask = int(math.ceil(len(word_starts) * p))
        if num_to_mask == 0:
            return tokens
        
        # Sample lengths for span masking
        if self.mask_span_distribution is not None:
            lengths = [self._sample_poisson() for _ in range(num_to_mask)]
            
            # Adjust lengths to fit available tokens
            total_length = sum(lengths)
            if total_length > len(word_starts):
                # Scale down lengths proportionally
                scale = len(word_starts) / total_length
                lengths = [max(1, int(l * scale)) for l in lengths]
            
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
                token_pos = pos + i
                if token_pos < len(result) - 1:  # Don't mask EOS
                    if random.random() < self.random_ratio:
                        # Replace with random token (avoid special tokens)
                        result[token_pos] = random.randint(
                            max(1, self.vocab_size // 10), 
                            self.vocab_size - 1
                        )
                    else:
                        # Replace with mask token
                        result[token_pos] = self.mask_token_id
                    
                    if self.replace_length == 0 and i > 0:
                        positions_to_remove.add(token_pos)
        
        # Remove positions if replace_length == 0
        if self.replace_length == 0:
            result = [token for i, token in enumerate(result) if i not in positions_to_remove]
        
        return result
    
    def add_insertion_noise(self, tokens: List[int], p: float) -> List[int]:
        """Add insertion noise to the sequence."""
        if p == 0.0 or len(tokens) <= 2:
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
                # Insert random token (avoid special tokens)
                new_token = random.randint(
                    max(1, self.vocab_size // 10), 
                    self.vocab_size - 1
                )
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
        middle = tokens[1:-1]
        if not middle:
            return tokens
            
        offset = random.randint(1, len(middle))
        
        # Rotate the middle part
        rotated_middle = middle[offset:] + middle[:offset]
        
        return [tokens[0]] + rotated_middle + [tokens[-1]]


class DataCollator:
    """
    A data collator that handles the collation of data for training.
    It can handle both single and multi-modal data with proper padding and attention masks.
    """
   
    def __init__(
        self, 
        tokenizer, 
        dae_processor: Optional[DAEProcessor] = None,
        padding: bool = True, 
        max_src_length: Optional[int] = None, 
        max_target_length: Optional[int] = None, 
        label_pad_token_id: int = -100,
        pad_to_max_length: bool = False,
        dae_ratio: float = 0.02,
        predict_with_generate: bool = False,
    ):
        self.tokenizer = tokenizer
        self.dae_processor = dae_processor
        self.dae_ratio = dae_ratio
        self.padding = padding
        self.max_src_length = max_src_length
        self.max_target_length = max_target_length
        self.label_pad_token_id = label_pad_token_id
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        self.pad_to_max_length = pad_to_max_length
        self.predict_with_generate = predict_with_generate

        # Validate pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0
    
    def pad(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates a list of features into a batch.
       
        Args:
            features: A list of feature dictionaries to collate.
       
        Returns:
            A dictionary containing the collated batch with proper padding and attention masks.
        """
        if not features:
            raise ValueError("Cannot collate empty list of features")

        batch_size = len(features)
        batch = {}
        
        # First pass: collect all keys and find max lengths
        all_keys = set()
        max_lengths = {}
        
        for feature in features:
            all_keys.update(feature.keys())
            
        for key in all_keys:
            max_lengths[key] = max(
                self._get_sequence_length(feature.get(key, [])) 
                for feature in features
            )
            
            if self.pad_to_max_length:
                max_lengths[key] = self.max_src_length if key in ['input_ids', 'attention_mask'] else self.max_target_length
            else:
                # Apply length limits if specified
                if key in ['input_ids', 'attention_mask'] and self.max_src_length:
                    max_lengths[key] = min(max_lengths[key], self.max_src_length)
                elif key == 'labels' and self.max_target_length:
                    max_lengths[key] = min(max_lengths[key], self.max_target_length)


        # Second pass: create tensors
        for key in all_keys:
            max_len = max_lengths[key]
            
            if key == 'labels':
                batch[key] = self._collate_labels(features, key, batch_size, max_len)
            else:
                batch[key], attention_mask = self._collate_sequence(
                    features, key, batch_size, max_len
                )
                
                # Create attention mask for input sequences
                if key.endswith('_ids') or key == 'input_ids':
                    mask_key = self._get_attention_mask_key(key)
                    batch[mask_key] = attention_mask
                    
        
        return batch
    
    def _get_sequence_length(self, value: Any) -> int:
        """Get the length of a sequence value."""
        if isinstance(value, (list, tuple)):
            return len(value)
        elif isinstance(value, torch.Tensor):
            return value.size(0) if value.dim() > 0 else 1
        elif isinstance(value, (int, float)):
            return 1
        else:
            return 0
    
    def _collate_labels(
        self, 
        features: List[Dict[str, Any]], 
        key: str, 
        batch_size: int, 
        max_len: int
    ) -> torch.Tensor:
        """Collate label sequences with proper padding."""
        tensor = torch.full(
            (batch_size, max_len), 
            self.label_pad_token_id, 
            dtype=torch.long
        )
        
        for i, feature in enumerate(features):
            value = feature.get(key)
            if value is None:
                continue
                
            length = min(self._get_sequence_length(value), max_len)
            if length > 0:
                tensor[i, :length] = self._convert_to_tensor(value, length)
        
        return tensor
    
    def _collate_sequence(
        self, 
        features: List[Dict[str, Any]], 
        key: str, 
        batch_size: int, 
        max_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate input sequences with padding and attention masks."""
        tensor = torch.full(
            (batch_size, max_len), 
            self.pad_token_id, 
            dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, feature in enumerate(features):
            value = feature.get(key)
            if value is None:
                continue
                
            length = min(self._get_sequence_length(value), max_len)
            if length > 0:
                tensor[i, :length] = self._convert_to_tensor(value, length)
                attention_mask[i, :length] = 1
        
        return tensor, attention_mask
    
    def _convert_to_tensor(self, value: Any, max_length: int) -> torch.Tensor:
        """Convert a value to a tensor with proper truncation."""
        if isinstance(value, list):
            # Truncate if necessary
            truncated_value = value[:max_length]
            return torch.tensor(truncated_value, dtype=torch.long)
        elif isinstance(value, torch.Tensor):
            # Ensure it's the right dtype and truncate if necessary
            tensor = value.to(dtype=torch.long)
            return tensor[:max_length] if tensor.dim() > 0 else tensor
        elif isinstance(value, (int, float)):
            return torch.tensor([int(value)], dtype=torch.long)
        else:
            raise ValueError(f"Unsupported type {type(value)} for conversion to tensor")
    
    def _get_attention_mask_key(self, key: str) -> str:
        """Generate the attention mask key name from the input key."""
        if key == 'input_ids':
            return 'attention_mask'
        elif key.endswith('_ids'):
            return key.replace('_ids', '_attention_mask')
        else:
            return f'{key}_attention_mask'
        
    def _expand_features(self, features: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[int]]:
        """Expand features for denoising autoencoder training."""
    
        # Expand features for input-target pairs
        expanded_features = []
        mse_mask = []
        
        for feature in features:
            if "input_ids" not in feature and "labels" not in feature:
                raise ValueError("Features must contain 'input_ids' and 'labels' for expansion", feature.keys())
            

            if "target_ids" not in feature:

                if self.dae_ratio > random.random():  # Apply DAE
                    mse_mask.append(0)
                    
                    # Apply noise to input_ids
                    if self.dae_processor:
                        noised_input_ids = self.dae_processor.noise(feature["input_ids"])
                    else:
                        noised_input_ids = feature["input_ids"]
                        
                    expanded_features.append({
                        "input_ids": noised_input_ids,
                        "labels": feature["input_ids"],
                    })
                    
                    # Apply noise to labels
                    if self.dae_processor:
                        noised_labels = self.dae_processor.noise(feature["labels"])
                    else:
                        noised_labels = feature["labels"]
                        
                    expanded_features.append({
                        "input_ids": noised_labels,
                        "labels": feature["labels"],
                    })
                
                else:  # Regular training
                    mse_mask.append(1)
                    expanded_features.append({
                        "input_ids": feature["input_ids"],
                        "labels": feature["labels"],
                    })
                    expanded_features.append({
                        "input_ids": feature["labels"],
                        "labels": feature["input_ids"],
                    })
                    

            else:
                # If "eval" is present, we treat it as evaluation data
                expanded_features.append({
                    "input_ids": feature["input_ids"],
                    "target_ids": feature["target_ids"],
                    "labels": feature["labels"],
                })
                # print("Regular training added")

        return expanded_features, mse_mask
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates a list of features into a batch.
       
        Args:
            features: A list of feature dictionaries to collate.
       
        Returns:
            A dictionary containing the collated batch with proper padding and attention masks.
        """
        
        if not features:
            raise ValueError("Cannot collate empty list of features")
        

        if self.dae_processor and not self.predict_with_generate:
            expanded_features, mse_mask = self._expand_features(features)
        else:
            expanded_features = [{"input_ids": f["input_ids"], "labels": f["labels"]} for f in features]
            mse_mask = []

        batch = self.pad(expanded_features)
        
        if len(mse_mask) > 0:
            batch["mse_mask"] = torch.tensor(mse_mask, dtype=torch.long)

        return batch


def main():
    """Test script for DAEProcessor and DataCollator."""
    print("Testing DAE Processor and Data Collator...")
    
    try:
        from transformers import AutoTokenizer
        
        # Initialize tokenizer (using a common model for testing)
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test data
        test_sentences = [
            "Hello world. This is a test sentence. How are you today?",
            "Machine learning is fascinating. It involves training models on data.",
            "The quick brown fox jumps over the lazy dog. This is a pangram.",
            "Natural language processing enables machines to understand human language.",
            "Artificial intelligence is transforming the way we interact with technology.",
            "The future of AI is promising and full of potential."
        ]
        
        # Tokenize test sentences
        tokenized_data = []
        for sentence in test_sentences:
            tokens = tokenizer(sentence, padding=False, truncation=True, max_length=128)
            input_ids = tokens["input_ids"]
            # Ensure input_ids is a list of integers
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.squeeze().tolist()
            elif isinstance(input_ids[0], list):
                input_ids = input_ids[0]  # Remove batch dimension if present
                
            tokenized_data.append({
                "input_ids": input_ids,
                "labels": input_ids  # For DAE, target is same as input
            })
        

        
        # Initialize DAE Processor
        dae_processor = DAEProcessor(
            tokenizer=tokenizer,
            mask_ratio=0.3,
            random_ratio=0.1,
            insert_ratio=0.1,
            rotate_ratio=0.2,
            permute_sentence_ratio=0.2,
            seed=42
        )
        


        
        # Initialize Data Collator
        data_collator = DataCollator(
            tokenizer=tokenizer,
            dae_processor=dae_processor,
            dae_ratio=0.5  # 50% chance of applying DAE
        )
        
        print(f"\nData Collator initialized with DAE ratio: {data_collator.dae_ratio}")
        
        # Test collation
        print(f"\nTesting batch collation:")
        batch = data_collator(tokenized_data)
        
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch shapes:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
            
        print(batch)
        
        print(f"\nFirst few input_ids from batch:")
        print(batch["input_ids"][:2, :10])  # Show first 2 samples, first 10 tokens
        
        print(f"\nFirst few labels from batch:")
        print(batch["labels"][:2, :10])  # Show first 2 samples, first 10 tokens
        
        print(f"\nMSE mask (0=DAE applied, 1=regular training):")
        print(batch["mse_mask"])
        
        print(batch["input_ids"])
        
        print("\n✓ All tests completed successfully!")
  
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()