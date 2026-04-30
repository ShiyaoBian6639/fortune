"""
Offline-first model loading for the QA system.

Mirrors the pattern in ``multimodal/text_encoder.py:ensure_local_model``.
After the one-time download, every load reads from a filesystem path with
``local_files_only=True`` — no HuggingFace network calls, no background
safetensors-conversion thread.

Models managed:
  BAAI/bge-m3   →  stock_data/models/bge-m3/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Silence transformers' INFO logs before first import.
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'stock_data' / 'models'

BGE_M3_LOCAL_DIR = MODELS_DIR / 'bge-m3'
BGE_M3_HF_ID     = 'BAAI/bge-m3'


def _find_hf_snapshot(hf_model_id: str) -> Optional[Path]:
    """Return path to existing local HF snapshot, or None if not cached."""
    try:
        import huggingface_hub
        path = huggingface_hub.snapshot_download(
            hf_model_id, local_files_only=True,
        )
        return Path(path)
    except Exception:
        return None


def ensure_bge_m3(local_dir: Path = BGE_M3_LOCAL_DIR) -> Path:
    """Idempotent: copy/download BAAI/bge-m3 to ``local_dir``.

    Returns the local directory path. After first call:
      - tokenizer + model can be loaded with ``local_files_only=True``
      - no network calls for any subsequent load
    """
    local_dir = Path(local_dir)
    if (local_dir / 'config.json').exists():
        return local_dir

    print(f"[local_loader] one-time bge-m3 setup → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _find_hf_snapshot(BGE_M3_HF_ID)
    if snapshot:
        load_src = str(snapshot)
        print(f"[local_loader] loading from HF cache: {snapshot}")
    else:
        load_src = BGE_M3_HF_ID
        print(f"[local_loader] downloading {BGE_M3_HF_ID} ...")

    tok = AutoTokenizer.from_pretrained(load_src,
                                         local_files_only=bool(snapshot))
    mdl = AutoModel.from_pretrained(load_src,
                                     local_files_only=bool(snapshot))

    tok.save_pretrained(str(local_dir))
    mdl.save_pretrained(str(local_dir))
    n = len(list(local_dir.iterdir()))
    print(f"[local_loader] saved {n} files to {local_dir}")
    return local_dir


if __name__ == '__main__':
    p = ensure_bge_m3()
    print(f"bge-m3 ready at {p}")
