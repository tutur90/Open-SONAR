#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm


class Margin(Enum):
    RATIO = "ratio"
    DISTANCE = "distance"
    ABSOLUTE = "absolute"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class XSim:
    """
    Compute the LASER-style xSIM metric using Hugging Face models.
    Reference: [https://github.com/facebookresearch/LASER](https://github.com/facebookresearch/LASER)
    """

    def __init__(
        self,
        min_sents: int = 0,
        index_comparison: bool = False,
        margin: str = Margin.ABSOLUTE.value,
        verbose: bool = False,
    ):
        self.min_sents = min_sents
        self.index_comparison = index_comparison
        self.margin = margin
        self.verbose = verbose

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

    def _calculate_error(
        self,
        x: np.ndarray,
        y: np.ndarray,
        margin: str,
        k: int,
    ) -> Tuple[int, int]:
        """Core xSIM error computation."""
        nbex = x.shape[0]

        # find best neighbour for every source vector
        closest_neighbor = self._score_knn(x, y, k, margin)   # (nbex, 1)
        
        ref = np.arange(nbex, dtype=int)
        err = nbex - np.equal(closest_neighbor.reshape(nbex), ref).sum()

        return err, nbex

    def xsim(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        margin: str = None,
        k: int = 4,
    ) -> Tuple[int, int]:
        """
        High-level wrapper for xSIM computation.
        
        Args:
            src: Source embeddings (np.ndarray)
            tgt: Target embeddings (np.ndarray)
            margin: Margin type override
            k: Number of nearest neighbors
        """
        margin = margin or self.margin
        
        if not Margin.has_value(margin):
            raise ValueError(f"Margin type: {margin} is not supported.")

        # Compute error
        return self._calculate_error(src, tgt, margin, k)

    def calc_xsim(
        self,
        embeddings: pd.DataFrame,
        src_langs: List[str] = None,
        tgt_langs: List[str] = None,
        xsimpp: bool = False,
        verbose: bool = False,
    ) -> List[List[str]]:
        """Pair-wise source-target evaluation – prints a table and returns results."""
        err_sum = 0
        tot_nbex = 0
        outputs = []

        all_langs = embeddings['lang'].unique().tolist()
        
        if src_langs is None:
            src_langs = all_langs
        if tgt_langs is None:
            tgt_langs = all_langs

        for (src_lang, tgt_lang) in itertools.product(src_langs, tgt_langs):
            if src_lang == tgt_lang:
                continue
                
            src_emb = embeddings[(embeddings['lang'] == src_lang) & 
                                 (embeddings['id'] >= 0)].sort_values('id')
            src_emb = np.stack(src_emb['emb'].values)

            tgt_emb = embeddings[embeddings['lang'] == tgt_lang].sort_values('id')
            
            if not xsimpp:
                tgt_emb = tgt_emb[tgt_emb['id'] >= 0]
            else:
                tgt_emb = pd.concat([
                    tgt_emb[tgt_emb['id'] >= 0].sort_values('id'),
                    tgt_emb[tgt_emb['id'] < 0].sort_values('id', ascending=False)
                ])
            
            tgt_emb = np.stack(tgt_emb['emb'].values)

            err, nbex = self._calculate_error(src_emb, tgt_emb, self.margin, 4)

            result = round(100 * err / nbex, 2)
            
            if nbex < self.min_sents:
                result = "skipped"
            else:
                err_sum += err
                tot_nbex += nbex

            outputs.append([f"{src_lang}-{tgt_lang}", f"{result}", f"{nbex}"])

        outputs.append([
            "average",
            f"{round(100 * err_sum / tot_nbex, 2)}",
            f"{len(outputs)}",
        ])
        
        if verbose or self.verbose:
            print(
                tabulate(
                    outputs,
                    tablefmt="psql",
                    headers=["src-tgt", f"xsim", "nbex"],
                )
            )
        return outputs

    def calc_xsim_nway(
        self, 
        embeddings: pd.DataFrame, 
        langs: List[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Compute an n-way error matrix (every language vs every other language)."""
        if langs is None:
            langs = embeddings['lang'].unique().tolist()
        
        err_matrix = np.zeros((len(langs), len(langs)))
        
        # Prepare embeddings for all languages
        lang_embeddings = {}
        for lang in tqdm(langs, desc="Preparing embeddings"):
            lang_emb = embeddings[(embeddings['lang'] == lang) & 
                                  (embeddings['id'] >= 0)].sort_values('id')
            lang_embeddings[lang] = np.stack(lang_emb['emb'].values)

        # Calculate pairwise errors
        for i1, src_lang in enumerate(langs):
            for i2, tgt_lang in enumerate(langs):
                if src_lang == tgt_lang:
                    err_matrix[i1, i2] = 0.0
                    continue
                
                src_emb = lang_embeddings[src_lang]
                tgt_emb = lang_embeddings[tgt_lang]
                
                err, nbex = self._calculate_error(src_emb, tgt_emb, self.margin, 4)
                err_matrix[i1, i2] = 100 * err / nbex

        # Create DataFrame with results
        df = pd.DataFrame(err_matrix, columns=langs, index=langs)
        
        # Add average row (excluding diagonal)
        avg_errors = []
        for i, lang in enumerate(langs):
            non_diag_errors = [err_matrix[i, j] for j in range(len(langs)) if i != j]
            avg_errors.append(np.mean(non_diag_errors))
        
        df.loc["avg"] = avg_errors

        if verbose or self.verbose:
            print(f"\n{tabulate(df, langs, floatfmt='.2f', tablefmt='grid')}\n")
            print(f"Global average: {np.mean(avg_errors):.2f}")
        
        return df
