# Remote / autodl deployment guide

For hosting the QA backend on an autodl GPU instance (or any cloud
machine with a small system disk + larger data disk). Tuned for the
common autodl layout:

- **System disk** `/root` ~30 GB — too small for HF cache + Python wheels
- **Data disk**   `/root/autodl-tmp` ~200 GB — where everything must live

## 1. Two env vars cover all download paths

```bash
# Where vLLM / transformers / huggingface_hub put downloaded model
# weights. Affects Qwen, bge-m3, anything pulled via from_pretrained.
export HF_HOME=/root/autodl-tmp/huggingface

# Where THIS project's `qa.local_loader` saves its local snapshot
# copies (currently just bge-m3's save_pretrained directory).
export QA_MODELS_DIR=/root/autodl-tmp/qa_models

# Optional: put pip / Python wheel cache on the data disk too — vLLM
# install pulls ~9 GB of CUDA wheels which would otherwise overflow /
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p $HF_HOME $QA_MODELS_DIR $PIP_CACHE_DIR $TMPDIR
```

Add these to `~/.bashrc` (or autodl's persistent profile) so they
survive container restarts.

## 2. Recommended directory layout on the data disk

```
/root/autodl-tmp/
├── huggingface/              # HF_HOME — Qwen + bge-m3 + others
│   ├── hub/
│   │   ├── models--Qwen--Qwen2.5-32B-Instruct-AWQ/  ~19 GB
│   │   └── models--BAAI--bge-m3/                     ~2.3 GB
│   └── ...
├── qa_models/                # QA_MODELS_DIR — bge-m3 save_pretrained copy
│   └── bge-m3/                                       ~2.3 GB
├── pip_cache/                # wheel cache
├── tmp/                      # tmpfile cache
├── tushare/                  # the repo, cloned here so stock_data/ is on data disk
│   ├── qa/
│   ├── stock_data/                                   varies (15-30 GB)
│   └── venv_vllm/            # vLLM venv lives here
└── ...
```

The repo itself should be cloned to `/root/autodl-tmp/tushare` so
`stock_data/` (which has `entities.faiss`, `news.faiss`, parquet files,
~15 GB) doesn't touch `/root`.

## 3. Disk budget on the data disk

| Item | Size |
|---|---|
| Qwen2.5-32B-Instruct-AWQ in HF cache | ~19 GB |
| bge-m3 in HF cache | 2.3 GB |
| bge-m3 local snapshot (save_pretrained) | 2.3 GB *(can be skipped — see §6)* |
| vLLM venv (torch + CUDA wheels + vLLM) | ~9 GB |
| Pip cache | ~5 GB |
| Repo data (`stock_data/qa/news.faiss` etc.) | ~12 GB |
| Repo data (other parquets / CSVs) | ~5–15 GB |
| **Total** | **~55–65 GB** |

Comfortably fits 200 GB with room for backtest caches.

## 4. Setup commands (one-time)

```bash
cd /root/autodl-tmp

# Clone repo to data disk
git clone <your-repo> tushare
cd tushare

# Create vLLM venv on data disk
python -m venv venv_vllm
source venv_vllm/bin/activate
pip install vllm openai requests fastapi uvicorn faiss-cpu pandas pyarrow

# Pre-download models (everything goes to $HF_HOME automatically)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ
huggingface-cli download BAAI/bge-m3

# Build local bge-m3 snapshot (goes to $QA_MODELS_DIR)
./venv_vllm/bin/python -m qa.local_loader

# Build/copy data files (entities.faiss, news.faiss, etc.) — usually
# rsync these from your laptop or rebuild via:
#   ./venv_vllm/bin/python -m qa.build_alias_dict
#   ./venv_vllm/bin/python -m qa.linker.predict
#   ./venv_vllm/bin/python -m qa.build_concept_tags
#   ./venv_vllm/bin/python -m qa.build_entity_index
#   ./venv_vllm/bin/python -m qa.build_news_index
```

## 5. Sanity-check the env vars are honoured

```bash
./venv_vllm/bin/python -c "
import os
print('HF_HOME       :', os.environ.get('HF_HOME'))
print('QA_MODELS_DIR :', os.environ.get('QA_MODELS_DIR'))
from qa.local_loader import MODELS_DIR, BGE_M3_LOCAL_DIR
print('MODELS_DIR    :', MODELS_DIR)
print('BGE_M3_LOCAL_DIR:', BGE_M3_LOCAL_DIR)
"
# All paths should be under /root/autodl-tmp
```

## 6. Optional: skip the bge-m3 local snapshot (save 2.3 GB)

`qa.local_loader.ensure_bge_m3` does a `save_pretrained` copy of
bge-m3 from the HF cache to `$QA_MODELS_DIR/bge-m3/`. The original
reason (avoiding the safetensors background-conversion thread) is a
transformers BERT-family quirk that doesn't affect bge-m3 in
practice. To skip the duplicate, point `QA_MODELS_DIR` at the HF
hub snapshot directly:

```bash
export QA_MODELS_DIR=$(./venv_vllm/bin/python -c "
import huggingface_hub as h
print(h.snapshot_download('BAAI/bge-m3', local_files_only=True).rsplit('/bge-m3', 1)[0])
")
```

This makes `BGE_M3_LOCAL_DIR` resolve to the HF cache snapshot —
`ensure_bge_m3` finds `config.json` already there and is a no-op.

## 7. Running the QA backend on vLLM

(Forthcoming — once `qa/rag/qa_engine.py` is refactored to talk to
vLLM via the OpenAI client, you'll start the stack as:)

```bash
# Terminal 1: vLLM serves Qwen at :8000
./venv_vllm/bin/vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --quantization awq_marlin \
    --port 8000 &

# Terminal 2: our FastAPI server at :8080 talks to vLLM
./venv_vllm/bin/python -m qa.api.server \
    --vllm-url http://localhost:8000/v1 \
    --port 8080 &
```
