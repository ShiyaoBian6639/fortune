"""
Dense (bge-m3) entity-candidate generator for the linker.

Given a batch of articles, returns top-K (entity_idx, cosine) hits per
article from the entity FAISS index. Used by ``qa/linker/predict.py`` in
``--use_dense`` mode to recover indirect mentions that the lexical
Aho-Corasick pass misses (e.g. "新能源车板块" → CATL/BYD when neither
name appears verbatim).

Caller is responsible for combining dense scores with lexical hits
into a final (article → list[ts_code]) decision.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd

from qa.rag.embedder import BgeM3Embedder


class DenseEntityMatcher:
    def __init__(self,
                 entity_faiss: str | Path = 'stock_data/qa/entities.faiss',
                 entity_meta:  str | Path = 'stock_data/qa/entities.parquet',
                 embedder: BgeM3Embedder | None = None,
                 max_length: int = 384):
        self.index = faiss.read_index(str(entity_faiss))
        self.meta  = pd.read_parquet(entity_meta)
        if embedder is None:
            embedder = BgeM3Embedder(max_length=max_length)
        self.embedder = embedder
        print(f"[dense_match] entity index ready  "
              f"({self.index.ntotal:,} vectors  meta={len(self.meta):,})")

    def search(self, article_texts: List[str], top_k: int = 20,
               batch_size: int = 32) -> List[List[Tuple[str, float]]]:
        """Return per-article list of (ts_code, cosine) tuples, length=top_k."""
        if not article_texts:
            return []
        vecs = self.embedder.encode(article_texts, batch_size=batch_size,
                                     show_progress=False)
        scores, idx = self.index.search(vecs.astype(np.float32), k=top_k)
        out: List[List[Tuple[str, float]]] = []
        for s_row, i_row in zip(scores, idx):
            row: List[Tuple[str, float]] = []
            for s, i in zip(s_row.tolist(), i_row.tolist()):
                if i < 0: continue
                row.append((self.meta.iloc[i]['ts_code'], float(s)))
            out.append(row)
        return out
