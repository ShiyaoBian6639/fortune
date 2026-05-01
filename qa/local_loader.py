"""
Offline-first model loading for the QA system.

Mirrors the pattern in ``multimodal/text_encoder.py:ensure_local_model``.
After the one-time download, every load reads from a filesystem path with
``local_files_only=True`` — no HuggingFace network calls, no background
safetensors-conversion thread.

Storage paths (configurable via env vars — important for cloud GPUs
like autodl where the system disk is small):

  HF_HOME             override for the HuggingFace cache (vLLM, Qwen,
                      transformers all read this). Set to a data-disk
                      path before running anything that downloads.
                      Default: ``~/.cache/huggingface/``

  QA_MODELS_DIR       override for THIS project's local model snapshots
                      (the save_pretrained copy of bge-m3). Default:
                      ``<repo>/stock_data/models/``.

Typical autodl setup (200 GB at /root/autodl-tmp):
    export HF_HOME=/root/autodl-tmp/huggingface
    export QA_MODELS_DIR=/root/autodl-tmp/qa_models

Models managed:
  BAAI/bge-m3   →  $QA_MODELS_DIR/bge-m3/
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

# Allow callers / cloud deployments to redirect the entire models
# directory to a data disk via env var, without code changes.
_DEFAULT_MODELS_DIR = ROOT / 'stock_data' / 'models'
MODELS_DIR = Path(os.environ.get('QA_MODELS_DIR', _DEFAULT_MODELS_DIR))

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


def ensure_bge_m3(local_dir: Path | None = None) -> Path:
    """Idempotent: copy/download BAAI/bge-m3 to ``local_dir``.

    Returns the local directory path. After first call:
      - tokenizer + model can be loaded with ``local_files_only=True``
      - no network calls for any subsequent load

    ``local_dir`` resolution order:
      1. explicit argument
      2. $QA_MODELS_DIR/bge-m3 (env var, picked up at module load)
      3. <repo>/stock_data/models/bge-m3 (default)
    """
    if local_dir is None:
        local_dir = BGE_M3_LOCAL_DIR
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
