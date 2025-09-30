#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import itertools
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import pandas as pd
from regex import T
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import sys
from tqdm import tqdm


sys.path.append(os.getcwd())
from transformers import AutoTokenizer
from open_sonar.text.models.modeling_sonar import SONARForText2Text
from open_sonar.nllb_langs import code_mapping



class Margin(Enum):
    RATIO = "ratio"
    DISTANCE = "distance"
    ABSOLUTE = "absolute"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class XSimEvaluator:
    """
    Compute the LASER-style xSIM metric using Hugging Face models.
    
    Example:
    >>> from transformers import AutoModel, AutoTokenizer
    >>> model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    >>> tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    >>> evaluator = XSimEvaluator(
    ...     base_dir=".",
    ...     corpus="mycorpus",
    ...     split="test",
    ...     src_encoder=model,
    ...     src_tokenizer=tokenizer,
    ... )
    >>> err, nbex, report = evaluator.xsim(
    ...     src=["Hello world!", "How are you?"],
    ...     tgt=["Bonjour le monde !", "Comment ça va ?"],
    ... )
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        corpus: str,
        split: str,
        src_encoder: "torch.nn.Module",
        src_tokenizer: "transformers.PreTrainedTokenizerBase",
        tgt_encoder: Optional["torch.nn.Module"] = None,
        tgt_tokenizer: Optional["transformers.PreTrainedTokenizerBase"] = None,
        min_sents: int = 0,
        index_comparison: bool = False,
        embedding_dimension: int = 1024,
        fp16: bool = False,
        margin: str = Margin.RATIO.value,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        # Configuration
        self.base_dir = Path(base_dir)
        self.corpus = corpus
        self.split = split
        self.min_sents = min_sents
        self.index_comparison = index_comparison
        self.emb_dimension = embedding_dimension
        self.fp16 = fp16
        self.margin = margin
        self.batch_size = batch_size
        self.verbose = verbose

        # Models and tokenizers
        if src_encoder is None:
            raise ValueError("Source encoder must be provided.")
        if src_tokenizer is None:
            raise ValueError("Source tokenizer must be provided.")
            
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder if tgt_encoder is not None else src_encoder
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer if tgt_tokenizer is not None else src_tokenizer

    def _score_margin(
        self,
        Dxy: np.ndarray,
        Ixy: np.ndarray,
        Ax: np.ndarray,
        Ay: np.ndarray,
        margin: str,
        k: int,
    ) -> np.ndarray:
        """Apply the chosen margin (ratio / distance) to the raw nearest-neighbour scores."""
        nbex = Dxy.shape[0]
        scores = np.zeros((nbex, k))
        for i in range(nbex):
            for j in range(k):
                jj = Ixy[i, j]
                a = Dxy[i, j]
                b = (Ax[i] + Ay[jj]) / 2
                if margin == Margin.RATIO.value:
                    scores[i, j] = a / b
                else:  # distance margin
                    scores[i, j] = a - b
        return scores

    def _score_knn(
        self, x: np.ndarray, y: np.ndarray, k: int, margin: str
    ) -> np.ndarray:
        """
        K-NN search using cosine similarity.
        Returns the index of the best neighbour for each row in x.
        """
        nbex, d = x.shape

        x_norm = normalize(x, norm="l2")
        y_norm = normalize(y, norm="l2")

        if margin == Margin.ABSOLUTE.value:
            sim = cosine_similarity(x_norm, y_norm)
            best_idx = np.argmax(sim, axis=1).reshape(-1, 1)
            return best_idx

        # top-k forward (x → y)
        sim_xy = cosine_similarity(x_norm, y_norm)
        top_k_idx_xy = np.argsort(sim_xy, axis=1)[:, -k:][:, ::-1]   # descending
        top_k_scores_xy = np.take_along_axis(sim_xy, top_k_idx_xy, axis=1)

        # top-k backward (y → x)
        sim_yx = cosine_similarity(y_norm, x_norm)
        top_k_idx_yx = np.argsort(sim_yx, axis=1)[:, -k:][:, ::-1]
        top_k_scores_yx = np.take_along_axis(sim_yx, top_k_idx_yx, axis=1)

        # average similarity per example (used for the margin)
        avg_xy = top_k_scores_xy.mean(axis=1)
        avg_yx = top_k_scores_yx.mean(axis=1)

        # apply margin
        scores = self._score_margin(
            top_k_scores_xy, top_k_idx_xy, avg_xy, avg_yx, margin, k
        )

        # pick neighbour with highest margin-adjusted score
        best = scores.argmax(axis=1)
        best_idx = np.zeros((nbex, 1), dtype=np.int32)
        for i in range(nbex):
            best_idx[i, 0] = top_k_idx_xy[i, best[i]]
        return best_idx

    def _get_transform(
        self, augmented_json: dict, closest_neighbor: str, src: str
    ) -> str:
        """Return the transformation type (or "Misaligned") for augmented evaluation."""
        if (
            closest_neighbor in augmented_json
            and augmented_json[closest_neighbor]["src"] == src
        ):
            return augmented_json[closest_neighbor]["errtype"]
        return "Misaligned"

    def _calculate_error(
        self,
        x: np.ndarray,
        y: np.ndarray,
        margin: str,
        k: int,
        eval_text: Union[str, None],
        augmented_json_path: Union[str, None],
    ) -> Tuple[int, int, Dict[str, int]]:
        """Core xSIM error computation."""
        if augmented_json_path:
            with open(augmented_json_path) as f:
                augmented_json = json.load(f)
            if x.shape[0] >= y.shape[0]:
                raise AssertionError(
                    f"Shape mismatch: source {x.shape[0]} >= target {y.shape[0]}"
                )
        else:
            augmented_json = None
            if x.shape != y.shape:
                raise AssertionError(
                    f"Shapes mismatch: source {x.shape} vs target {y.shape}"
                )

        nbex = x.shape[0]
        augmented_report: Dict[str, int] = {}

        # find best neighbour for every source vector
        closest_neighbor = self._score_knn(x, y, k, margin)   # (nbex, 1)

        if eval_text:   # textual error (line-by-line comparison)
            lines = open(eval_text, encoding="utf-8", errors="surrogateescape").readlines()
            err = 0
            for ex in range(nbex):
                if lines[ex] != lines[closest_neighbor[ex, 0]]:
                    err += 1
                    if augmented_json:
                        transform = self._get_transform(
                            augmented_json,
                            lines[closest_neighbor[ex, 0]].strip(),
                            lines[ex].strip(),
                        )
                        augmented_report[transform] = augmented_report.get(transform, 0) + 1
        else:           # pure index-based error
            ref = np.arange(nbex, dtype=int)
            err = nbex - np.equal(closest_neighbor.reshape(nbex), ref).sum()

        return err, nbex, augmented_report

    @staticmethod
    def _mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> np.ndarray:
        """
        Simple mean-pooling (ignore padding tokens). Returns a 2-D NumPy array
        with shape (batch, hidden_dim).
        """
        import torch

        # mask -> (batch, seq_len, 1) for broadcasting
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = masked_embeddings.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)   # avoid division by zero
        mean_pooled = summed / lengths
        return mean_pooled.cpu().numpy()

    def _embed_with_hf(
        self,
        sentences: List[str],
        model: "torch.nn.Module",
        tokenizer: "transformers.PreTrainedTokenizerBase",
        lang_code: Optional[str] = None,
    ) -> np.ndarray:
        """
        Turn a list of raw sentences into a NumPy matrix of sentence embeddings.
        Sets tokenizer.src_lang if lang_code is provided and tokenizer supports it.
        """
        import torch
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set source language if supported by tokenizer
        if lang_code and hasattr(tokenizer, 'src_lang'):
            if self.verbose:
                print(f"Setting tokenizer.src_lang to: {lang_code}")
            tokenizer.src_lang = lang_code

        all_embeddings = []

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            # tokenization
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(tokenizer, 'model_max_length', 512),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=False, return_dict=False)
                
                # Handle different output formats
                if isinstance(out, tuple):
                    hidden_states = out[0]  # (batch, seq_len, hidden_dim)
                else:
                    hidden_states = out.last_hidden_state
                
                # Use mean pooling if sequence length > 1, otherwise squeeze
                if hidden_states.shape[1] > 1:
                    embeddings = self._mean_pool(hidden_states, inputs["attention_mask"])
                else:
                    embeddings = hidden_states.squeeze().cpu().numpy()
                    if embeddings.ndim == 1:
                        embeddings = embeddings.reshape(1, -1)
                
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)   # (N, hidden_dim)

    def xsim(
        self,
        src: Union[np.ndarray, List[str]],
        tgt: Union[np.ndarray, List[str]],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        margin: str = None,
        k: int = 4,
        eval_text: str = None,
        augmented_json: str = None,
    ) -> Tuple[int, int, Dict[str, int]]:
        """
        High-level wrapper for xSIM computation.
        
        Args:
            src: Source data (np.ndarray or list of sentences)
            tgt: Target data (np.ndarray or list of sentences)
            src_lang: Source language code (for tokenizer)
            tgt_lang: Target language code (for tokenizer)
            margin: Margin type override
            k: Number of nearest neighbors
            eval_text: Path to evaluation text file
            augmented_json: Path to augmented JSON file
        """
        margin = margin or self.margin
        
        if not Margin.has_value(margin):
            raise ValueError(f"Margin type: {margin} is not supported.")

        # Create source embeddings
        if isinstance(src, np.ndarray):
            src_emb = src
        else:  # list of sentences
            src_emb = self._embed_with_hf(src, self.src_encoder, self.src_tokenizer, src_lang)

        # Create target embeddings
        if isinstance(tgt, np.ndarray):
            tgt_emb = tgt
        else:  # list of sentences
            tgt_emb = self._embed_with_hf(tgt, self.tgt_encoder, self.tgt_tokenizer, tgt_lang)

        # Compute error
        return self._calculate_error(
            src_emb,
            tgt_emb,
            margin,
            k,
            eval_text,
            augmented_json,
        )

    def _embed_language(
        self,
        lang: str,
        tmpdir: str,
        tgt_aug_langs: List[str] = None,
        _type: str = "source",
        overwrite: bool = False
    ) -> Tuple[str, str, str, Optional[str]]:
        """
        Embed a single language, handling augmented data if needed.
        Returns (lang, input_file, output_file, augmented_json)
        """
        if tgt_aug_langs is None:
            tgt_aug_langs = []

        # File paths
        augjson = None
        fname = f"{lang}.{self.split}"
        infile = self.base_dir / self.corpus / self.split / fname
        if not infile.is_file():
            raise FileNotFoundError(f"{infile} does not exist")
        outfile = Path(tmpdir) / _type /  f"{lang}.emb"
        

        if outfile.parent and not outfile.parent.is_dir():
            outfile.parent.mkdir(parents=True)

        # Handle augmented target data
        if lang in tgt_aug_langs:
            aug_fname = f"{lang}_augmented.{self.split}"
            aug_json_name = f"{lang}_errtype.{self.split}.json"
            aug_dir = self.base_dir / self.corpus / (self.split + "_augmented")
            augjson = aug_dir / aug_json_name
            auginfile = aug_dir / aug_fname

            for p in (augjson, auginfile):
                if not p.is_file():
                    raise FileNotFoundError(str(p))

            # Combine original and augmented data
            combined = Path(tmpdir) / f"combined_{lang}"
            with open(combined, "w", encoding="utf-8") as out_f:
                for f in (infile, auginfile):
                    with open(f, encoding="utf-8") as in_f:
                        out_f.write(in_f.read())
            infile = combined

        # Read sentences and embed
        with open(infile, encoding="utf-8") as f:
            sentences = [ln.strip() for ln in f.readlines() if ln.strip()]

        # Determine which encoder/tokenizer to use
        is_source = True  # We'll use source encoder by default, can be made configurable
        encoder = self.src_encoder if is_source else self.tgt_encoder
        tokenizer = self.src_tokenizer if is_source else self.tgt_tokenizer

        if outfile.is_file() and not overwrite:
            if self.verbose:
                print(f"Embedding for {lang} already exists: {outfile}")

        else: 
        # Generate embeddings
            hf_emb = self._embed_with_hf(sentences, encoder, tokenizer, lang)
        
        # Write to .bin file
            hf_emb.astype(np.float32).tofile(str(outfile))

        if not (outfile.is_file() and outfile.stat().st_size > 0):
            raise RuntimeError(f"Error encoding {infile}")

        return lang, str(infile), str(outfile), str(augjson) if augjson else None

    def calc_xsim(
        self,
        embdir: str,
        src_langs: List[str],
        tgt_langs: List[str],
        tgt_aug_langs: List[str] = None,
        overwrite: bool = False 
    ) -> None:
        """Pair-wise source-target evaluation – prints a table and optional aug-report."""
        if tgt_aug_langs is None:
            tgt_aug_langs = []

        err_sum = 0
        tot_nbex = 0
        outputs = []

        # Embed all languages
        # print("Embedding source languages...")
        src_emb_data = []
        for lang in tqdm(src_langs, desc="Source languages"):
            src_emb_data.append(self._embed_language(lang, embdir, overwrite=overwrite, _type="source"))

        print("Embedding target languages...")
        tgt_emb_data = []
        for lang in tqdm(tgt_langs, desc="Target languages"):
            tgt_emb_data.append(self._embed_language(lang, embdir, tgt_aug_langs, overwrite=overwrite, _type="target"))

        aug_report_by_tgt = defaultdict(dict)

        for (src_lang, _, src_emb, _), (tgt_lang, tgt_txt, tgt_emb, augjson) in itertools.product(
            src_emb_data, tgt_emb_data
        ):
            if src_lang == tgt_lang:
                continue

            if self.verbose:
                print(f"Calculating xSIM for: {src_lang} → {tgt_lang}")

            # Load embeddings from files
            src_emb_array = np.fromfile(src_emb, dtype=np.float32).reshape(-1, self.emb_dimension)
            tgt_emb_array = np.fromfile(tgt_emb, dtype=np.float32).reshape(-1, self.emb_dimension)

            err, nbex, aug_report = self._calculate_error(
                src_emb_array,
                tgt_emb_array,
                self.margin,
                4,  # k
                tgt_txt if not self.index_comparison else None,
                augjson,
            )

            result = round(100 * err / nbex, 2)
            if tgt_lang in tgt_aug_langs:
                aug_report_by_tgt[tgt_lang][src_lang] = aug_report

            if nbex < self.min_sents:
                result = "skipped"
            else:
                err_sum += err
                tot_nbex += nbex

            outputs.append([self.corpus, f"{src_lang}-{tgt_lang}", f"{result}", f"{nbex}"])

        outputs.append(
            [
                self.corpus,
                "average",
                f"{round(100 * err_sum / tot_nbex, 2)}",
                f"{len(outputs)}",
            ]
        )
        
        print(
            tabulate(
                outputs,
                tablefmt="psql",
                headers=["dataset", "src-tgt", f"xsim{'(++)' if tgt_aug_langs else ''}", "nbex"],
            )
        )

        # Print augmented reports
        for tgt_aug_lang in tgt_aug_langs:
            df = pd.DataFrame.from_dict(aug_report_by_tgt[tgt_aug_lang]).fillna(0).T
            print(f"\nAbsolute error under augmented transformations for: {tgt_aug_lang}")
            print(tabulate(df, df.columns, floatfmt=".2f", tablefmt="grid"))

    def calc_xsim_nway(self, embdir: str, langs: List[str]) -> None:
        """Compute an n-way error matrix (every language vs every other language)."""
        err_matrix = np.zeros((len(langs), len(langs)))
        
        print("Embedding all languages...")
        emb_data = []
        for lang in tqdm(langs, desc="Languages"):
            emb_data.append(self._embed_language(lang, embdir))

        for i1, (src_lang, _, src_emb, _) in enumerate(emb_data):
            for i2, (tgt_lang, tgt_txt, tgt_emb, _) in enumerate(emb_data):
                if src_lang == tgt_lang:
                    continue
                
                # Load embeddings from files
                src_emb_array = np.fromfile(src_emb, dtype=np.float32).reshape(-1, self.emb_dimension)
                tgt_emb_array = np.fromfile(tgt_emb, dtype=np.float32).reshape(-1, self.emb_dimension)
                
                err, nbex, _ = self._calculate_error(
                    src_emb_array,
                    tgt_emb_array,
                    self.margin,
                    4,  # k
                    tgt_txt,
                    None,
                )
                err_matrix[i1, i2] = 100 * err / nbex

        df = pd.DataFrame(err_matrix, columns=langs, index=langs)
        df.loc["avg"] = df.sum() / float(df.shape[0] - 1)   # exclude diagonal
        print(f"\n{tabulate(df, langs, floatfmt='.2f', tablefmt='grid')}\n")
        print(f"Global average: {df.loc['avg'].mean():.2f}")


def main():
    """Example usage with SONAR model"""
    # Load SONAR model and tokenizer
    
    pretrained_model_path = ""

    
    hf_model = SONARForText2Text.from_pretrained(
        pretrained_model_path
    ).model.encoder
    hf_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")

    print("model max length:", hf_tokenizer.model_max_length)
    print(pretrained_model_path)

    evaluator = XSimEvaluator(
        base_dir=".",
        corpus="datasets/flores200",
        split="devtest",
        src_encoder=hf_model,
        src_tokenizer=hf_tokenizer,
        min_sents=1,
        embedding_dimension=1024,  # SONAR embedding dimension
        fp16=True,
        margin=Margin.ABSOLUTE.value,
        batch_size=128,
        verbose=False,
    )

    # Create embeddings directory
    embdir = "data/embeddings/sonar_mini_"
    os.makedirs(embdir, exist_ok=True)

    # Run evaluation
    print("\n=== Running calc_xsim ===\n")
    evaluator.calc_xsim(
        embdir,
        src_langs=list(code_mapping.values()),  # source languages from NLLB code mapping
        overwrite=True,  # overwrite existing embeddings
        tgt_langs=["eng_Latn"],  # target language
        tgt_aug_langs=["eng_Latn"]  # augmented languages
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()