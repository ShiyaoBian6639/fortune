"""
Thin bge-m3 dense-embedding helper.

Exposes a single ``BgeM3Embedder`` class that:
  - Loads BAAI/bge-m3 from the local snapshot built by ``qa.local_loader``
  - Encodes a list of strings into L2-normalised float32 vectors of dim 1024
  - Uses CLS pooling (the bge-m3 dense head is the [CLS] token)
  - Supports fp16 inference on CUDA for ~2× throughput

Why not FlagEmbedding? — that wrapper bundles dense + sparse + colbert and
pulls in its own tokenizer quirks. We only need the dense head, and the
HuggingFace AutoModel pathway gives us the same vectors with one less
dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from qa.local_loader import ensure_bge_m3


class BgeM3Embedder:
    def __init__(self, local_dir: Path | None = None,
                 device: str | None = None,
                 fp16: bool = True,
                 max_length: int = 512):
        local_dir = Path(local_dir) if local_dir else ensure_bge_m3()
        self.tok = AutoTokenizer.from_pretrained(str(local_dir),
                                                  local_files_only=True)
        self.model = AutoModel.from_pretrained(str(local_dir),
                                                local_files_only=True)
        self.model.eval()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.dtype  = torch.float16 if (fp16 and device == 'cuda') else torch.float32
        self.model.to(device=device, dtype=self.dtype)

        self.max_length = max_length
        self.dim = int(self.model.config.hidden_size)   # 1024 for bge-m3
        print(f"[embedder] bge-m3 ready  device={device}  dtype={self.dtype}  "
              f"dim={self.dim}  max_len={max_length}")

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        """Return (N, dim) L2-normalised float32 vectors."""
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        if show_progress:
            from tqdm import tqdm
            it = tqdm(range(0, len(texts), batch_size),
                      total=(len(texts) + batch_size - 1) // batch_size,
                      desc='[embed]', mininterval=2.0)
        else:
            it = range(0, len(texts), batch_size)
        for i in it:
            batch = list(texts[i: i + batch_size])
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length,
                           return_tensors='pt').to(self.device)
            h = self.model(**enc).last_hidden_state[:, 0]   # CLS pool
            h = F.normalize(h.float(), p=2, dim=-1)
            out[i: i + len(batch)] = h.cpu().numpy()
        return out
