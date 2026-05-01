# Remote / autodl deployment guide

For hosting the QA backend on an autodl GPU instance (or any cloud
machine with a small system disk + larger data disk). Tuned for the
common autodl layout:

- **System disk** `/root` ~30 GB — too small for HF cache + Python wheels
- **Data disk**   `/root/autodl-tmp` ~200 GB — where everything must live

## TL;DR — what you must transfer before running

After §1–§4 (env vars, venv, model downloads), you still need
**~12 GB of pre-built data** under `stock_data/` for the QA backend
to answer anything. The minimum required from your local machine to
the remote `:/root/autodl-tmp/tushare/`:

| Path | Size | Why |
|---|---|---|
| `stock_data/qa/aliases.json` | ~3 MB | Stock alias dict (5,190 stocks) |
| `stock_data/qa/news_linked.parquet` | ~134 MB | Articles → ts_codes |
| `stock_data/qa/entities.faiss` | ~22 MB | Per-stock embeddings |
| `stock_data/qa/entities.parquet` | ~1 MB | Sidecar for entities.faiss |
| `stock_data/qa/news.faiss` | ~8.5 GB | News article embeddings |
| `stock_data/qa/news_meta.parquet` | ~70 MB | Sidecar for news.faiss |
| `stock_data/qa/concept_tags.json` | ~0.6 MB | News-derived concept tags |
| `stock_data/qa/company_business.csv` | ~5 MB | Tushare main_business |
| `stock_data/qa/linker_reranker.pkl` | small | Optional dense reranker |
| `stock_data/sh/<code>.csv` × 2,400 | ~1.8 GB | SH daily prices |
| `stock_data/sz/<code>.csv` × 3,000 | ~2.2 GB | SZ daily prices |
| `stock_data/fina_indicator/*.csv` × 5,200 | ~150 MB | Quarterly fundamentals |
| `stock_data/static_features/*.csv` | ~10 MB | Company info / index members |
| `stock_data/stock_list.csv` + `stock_sectors.csv` | <1 MB | Stock universe |
| `stock_data/news_sentiment_qwen.csv` | ~3 MB | Sentiment trend |
| **Subtotal** | **~13 GB** | |

Optional (only if you'll re-link the news corpus on the remote, otherwise skip):

| Path | Size | Skip if you have `news_linked.parquet` |
|---|---|---|
| `stock_data/news_corpus_dedup.parquet` | ~1.2 GB | yes |
| `stock_data/news_corpus_dedup_codes.parquet` | ~70 MB | yes |

Plus the model weights downloaded on the remote (§4):
- Qwen2.5-32B-Instruct-AWQ in HF cache (~19 GB)
- bge-m3 in HF cache (~2.3 GB)

**Total disk on data volume: ~35 GB. The 200 GB disk has plenty of headroom.**

Concrete rsync command (run from your **local** repo root):

```bash
HOST=region-1.autodl.com    # ← your autodl host
PORT=22000                  # ← your autodl ssh port

rsync -avzP -e "ssh -p $PORT" \
    stock_data/qa/ stock_data/sh/ stock_data/sz/ \
    stock_data/fina_indicator/ stock_data/static_features/ \
    stock_data/stock_list.csv stock_data/stock_sectors.csv \
    stock_data/news_sentiment_qwen.csv \
    root@$HOST:/root/autodl-tmp/tushare/stock_data/
```

The `-z` compresses over the wire (parquet + CSV compress well), `-P`
shows progress and resumes if interrupted. Expect 10–30 min depending
on bandwidth.

After transfer, run §7's verifier to confirm everything landed.

## 1. Two env vars cover all download paths

`export` is **per-session** by default — the variables vanish when you
log out or the instance restarts. To make them permanent, append the
block once to `~/.bashrc` (which is on the persistent `/root` volume
on autodl, so it survives stop/start cycles):

```bash
cat >> ~/.bashrc <<'EOF'

# QA backend storage paths (persist across autodl restarts)
export HF_HOME=/root/autodl-tmp/huggingface
export QA_MODELS_DIR=/root/autodl-tmp/qa_models
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$HF_HOME" "$QA_MODELS_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
EOF

# Apply to the current shell without logging out
source ~/.bashrc
```

What each one does:

| Var | Read by | Default if unset |
|---|---|---|
| `HF_HOME` | vLLM / transformers / huggingface_hub — Qwen, bge-m3, anything via `from_pretrained` | `~/.cache/huggingface` |
| `QA_MODELS_DIR` | This project's `qa.local_loader` — bge-m3 `save_pretrained` copy | `<repo>/stock_data/models/` |
| `PIP_CACHE_DIR` | pip wheel cache — vLLM install pulls ~9 GB of CUDA wheels | `~/.cache/pip` |
| `TMPDIR` | Linux temp files (build steps, large extracts) | `/tmp` |

After the one-time append, every new shell — including the one autodl
gives you on instance restart — picks them up automatically. Verify
on a fresh login:

```bash
echo $HF_HOME       # should print /root/autodl-tmp/huggingface
echo $QA_MODELS_DIR # should print /root/autodl-tmp/qa_models
```

If either is empty, your `.bashrc` didn't apply — log out and back in
or `source ~/.bashrc` manually.

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

# Pre-download models (everything goes to $HF_HOME automatically).
# huggingface_hub ≥ 0.27 renamed `huggingface-cli` → `hf`. Use whichever
# is available on your image:
hf download Qwen/Qwen2.5-32B-Instruct-AWQ      # newer (autodl 2024+ images)
hf download BAAI/bge-m3
# or, on older boxes:
# huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ
# huggingface-cli download BAAI/bge-m3

# Build local bge-m3 snapshot (goes to $QA_MODELS_DIR)
python -m qa.local_loader
```

After this, the venv + models are on disk, but `stock_data/` is still
empty. Pick **one** of the next two options to populate it.

## 5. Get the data files

The QA backend needs ~30 GB of data files under `stock_data/` —
half are raw Tushare pulls, half are pre-built FAISS indexes /
parquets. Two paths to get them:

### Option A — rsync from your local machine (fastest, ~30 min)

If you already have everything built locally (the case after running
the QA pipeline on your laptop), just transfer the files. From your
local repo:

```bash
HOST=region-1.autodl.com    # your autodl host
PORT=22000                  # your autodl ssh port
DEST=root@$HOST:/root/autodl-tmp/tushare/stock_data/
SSH="ssh -p $PORT"

# Required minimum for QA inference (~13 GB)
rsync -avzP -e "$SSH" stock_data/qa/                $DEST/qa/
rsync -avzP -e "$SSH" stock_data/sh/                $DEST/sh/
rsync -avzP -e "$SSH" stock_data/sz/                $DEST/sz/
rsync -avzP -e "$SSH" stock_data/fina_indicator/    $DEST/fina_indicator/
rsync -avzP -e "$SSH" stock_data/static_features/   $DEST/static_features/
rsync -avzP -e "$SSH" stock_data/stock_list.csv \
                       stock_data/stock_sectors.csv \
                       stock_data/news_sentiment_qwen.csv  $DEST

# Optional: only if you plan to RE-RUN the linker on the remote
# (otherwise news_linked.parquet alone is enough for QA inference).
rsync -avzP -e "$SSH" stock_data/news_corpus_dedup.parquet \
                       stock_data/news_corpus_dedup_codes.parquet  $DEST
```

`-z` compresses over the wire — important for parquet/CSV which
compress well. `-P` shows progress and resumes interrupted transfers.

### Option B — fresh fetch + rebuild on the remote (clean, ~6–8 hr)

Only needed if you don't have the data locally. The remote machine
has Tushare access from US.

```bash
# 1. Configure Tushare token (one-time)
# Edit dl/config.py and paste your Tushare Pro token in TUSHARE_TOKEN.

# 2. Pull stock daily bars (forward + backward fill, ~30 min)
python extend_stock_data.py --update --workers 8

# 3. Pull static features (~5 min)
python -m api.static_features --source stock_company
python -m api.static_features --source index_member_pit

# 4. Pull fundamentals (~10 min, fina_indicator across all stocks)
python -m api.fina_extras   # whatever the entry point is

# 5. Pull news corpus (~3-5 hr — the long pole)
python main.py              # runs features.news.run('download')

# 6. Pull main_business + business_scope sidecar (~30 sec)
python -m qa.fetch_company_business

# 7. Build all QA-derived artifacts in order:
python -m qa.build_alias_dict       # ~5 sec
python -m qa.linker.predict          # ~1 min over 1.95 M articles
python -m qa.build_concept_tags      # ~30 sec
python -m qa.build_entity_index      # ~30 sec on GPU
python -m qa.linker.train_reranker   # ~5 min
python -m qa.linker.predict --use_dense  # ~80 min, optional
python -m qa.build_news_index        # ~110 min on RTX 4070;
                                                     # ~50 min on RTX 5090
```

Steps 5 and 7-final (news index) are the slow ones. If you don't need
free-text news semantic search at first, skip step 7-final and the
QA backend still works (alias + entity-semantic paths cover most
queries — 75 % of the demo passes without it).

## 6. What ends up under stock_data/ — file-by-file

After either option, you should have:

| Path | Size | Purpose |
|---|---|---|
| `stock_data/sh/<code>.csv` × 2,400 | ~1.8 GB | SH daily price bars (price summary in QA context) |
| `stock_data/sz/<code>.csv` × 3,000 | ~2.2 GB | SZ daily price bars |
| `stock_data/fina_indicator/*.csv` × 5,200 | ~150 MB | Quarterly fundamentals (EPS / ROE / margins) |
| `stock_data/static_features/stock_company.csv` | ~1 MB | Company info (chairman / manager / sector) |
| `stock_data/static_features/index_member_pit.csv` | ~5 MB | CSI300 / SSE50 membership over time |
| `stock_data/stock_list.csv` | <1 MB | Stock universe |
| `stock_data/stock_sectors.csv` | <1 MB | Shenwan sector taxonomy |
| `stock_data/news_sentiment_qwen.csv` | ~3 MB | Sentiment trend (used in entity card context) |
| `stock_data/news_corpus_dedup.parquet` | ~1.2 GB | Raw deduped news corpus *(only needed for re-linking)* |
| `stock_data/news_corpus_dedup_codes.parquet` | ~70 MB | Regex-confirmed positives *(only needed for reranker training)* |
| **`stock_data/qa/aliases.json`** | ~3 MB | **Alias dict — required** |
| **`stock_data/qa/news_linked.parquet`** | ~134 MB | **Linker output — required** |
| **`stock_data/qa/entities.faiss`** + `entities.parquet` | ~25 MB | **Entity index — required** |
| `stock_data/qa/concept_tags.json` | ~0.6 MB | News-derived concept tags |
| `stock_data/qa/company_business.csv` | ~5 MB | Tushare main_business (preserved for future use) |
| `stock_data/qa/linker_reranker.pkl` | small | Optional dense reranker |
| **`stock_data/qa/news.faiss`** + `news_meta.parquet` | ~8.5 GB | **News index — required for free-text article search** |

The four bolded files are the must-have minimum if you only want
inference. Everything else is either a build input (raw data) or
optional enhancement.

## 7. Verify the data is in place

```bash
python -c "
from pathlib import Path
import pandas as pd, json

DATA = Path('stock_data')
checks = [
  ('aliases',          DATA/'qa/aliases.json',          'json',    5000),
  ('news_linked',      DATA/'qa/news_linked.parquet',   'parquet', 400_000),
  ('entities.faiss',   DATA/'qa/entities.faiss',        'exists',  None),
  ('entities.parquet', DATA/'qa/entities.parquet',      'parquet', 5000),
  ('news.faiss',       DATA/'qa/news.faiss',            'exists',  None),
  ('news_meta',        DATA/'qa/news_meta.parquet',     'parquet', 1_500_000),
  ('fina_indicator',   DATA/'fina_indicator',           'dir',     5000),
  ('sh prices',        DATA/'sh',                       'dir',     2000),
  ('sz prices',        DATA/'sz',                       'dir',     2500),
]
for name, p, kind, min_n in checks:
  if not p.exists():
    print(f'  MISSING {name}: {p}')
    continue
  if kind == 'json':
    n = len(json.loads(p.read_text(encoding='utf-8')))
  elif kind == 'parquet':
    n = len(pd.read_parquet(p))
  elif kind == 'dir':
    n = len(list(p.glob('*')))
  else:
    n = p.stat().st_size
  ok = '[ok]' if (min_n is None or n >= min_n) else '[low]'
  print(f'  {ok} {name:<18} count/size = {n:>12,}')
"
```

Expected output:
```
  [ok] aliases             count/size =        5,190
  [ok] news_linked         count/size =      548,231
  [ok] entities.faiss      count/size =   23,789,632
  [ok] entities.parquet    count/size =        5,190
  [ok] news.faiss          count/size = 8,545,239,316
  [ok] news_meta           count/size =    1,951,451
  [ok] fina_indicator      count/size =        5,200
  [ok] sh prices           count/size =        2,454
  [ok] sz prices           count/size =        3,077
```

## 8. Sanity-check the env vars are honoured

```bash
python -c "
import os
print('HF_HOME       :', os.environ.get('HF_HOME'))
print('QA_MODELS_DIR :', os.environ.get('QA_MODELS_DIR'))
from qa.local_loader import MODELS_DIR, BGE_M3_LOCAL_DIR
print('MODELS_DIR    :', MODELS_DIR)
print('BGE_M3_LOCAL_DIR:', BGE_M3_LOCAL_DIR)
"
# All paths should be under /root/autodl-tmp
```

## 9. Optional: skip the bge-m3 local snapshot (save 2.3 GB)

`qa.local_loader.ensure_bge_m3` does a `save_pretrained` copy of
bge-m3 from the HF cache to `$QA_MODELS_DIR/bge-m3/`. The original
reason (avoiding the safetensors background-conversion thread) is a
transformers BERT-family quirk that doesn't affect bge-m3 in
practice. To skip the duplicate, point `QA_MODELS_DIR` at the HF
hub snapshot directly:

```bash
export QA_MODELS_DIR=$(python -c "
import huggingface_hub as h
print(h.snapshot_download('BAAI/bge-m3', local_files_only=True).rsplit('/bge-m3', 1)[0])
")
```

This makes `BGE_M3_LOCAL_DIR` resolve to the HF cache snapshot —
`ensure_bge_m3` finds `config.json` already there and is a no-op.

## 10. Running the QA backend on vLLM

`qa.rag.qa_engine.QAEngine` is now a thin OpenAI-client wrapper around
vLLM. Two-process stack:

```bash
# ─── Terminal 1 (or screen window) — vLLM serves Qwen at :8000 ───
screen -S vllm
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --quantization awq_marlin \
    --port 8000
# Ctrl+A then D to detach. Reattach with: screen -r vllm

# ─── Terminal 2 — our FastAPI server at :8080 ───
screen -S api
python -m qa.api.server \
    --vllm-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --host 0.0.0.0 --port 8080
# Ctrl+A then D to detach.
```

`--host 0.0.0.0` exposes on the LAN — needed if you're hitting the API
from your laptop. autodl typically ssh-forwards :8080.

Verify both:

```bash
curl -s http://localhost:8080/healthz
# {"status": "ok"}

curl -s http://localhost:8080/vllm_health
# {"status": "ok", "vllm_url": "http://localhost:8000/v1",
#  "served_models": ["Qwen/Qwen2.5-32B-Instruct-AWQ"], "cache_hits": 0}

curl -s -X POST http://localhost:8080/ask \
     -H "Content-Type: application/json" \
     --data '{"query":"600519.SH 业绩","top_k":5}' | python -c \
     "import json, sys; d=json.load(sys.stdin); print(d['answer'][:300])"
```

The first request takes ~3–6 s (vLLM warm), subsequent identical
queries return instantly from the LRU cache. Concurrent users now
share KV cache via vLLM's PagedAttention — no GPU lock.

### Re-running the 50-question demo against vLLM

```bash
python -m qa.eval.run_phase2_demo \
    --api http://localhost:8080 \
    --out stock_data/qa/phase2_demo_report_vllm.json
```

Expected on Qwen2.5-32B-AWQ + vLLM: substantially fewer rep loops,
better synthesis quality, ~3 s per question avg vs ~10 s on the 7B
int4. Re-enable `main_business` in `qa/build_entity_index.py` first
— the v4/v5 regression on 7B doesn't recur with the bigger model.

## 11. Build derived data ON the VM (skip the 8.5 GB upload)

If your upload bandwidth is slow, transfer **only the build inputs**
and let the 5090 rebuild the heavy outputs (`news.faiss`,
`news_linked.parquet`, `entities.faiss`, etc.) locally. Net savings:
~10 GB of upload, ~50 min of GPU compute.

### What to upload

| Path | Size | Why |
|---|---|---|
| `stock_data/news_corpus_dedup.parquet` | ~1.2 GB | input to linker.predict + build_news_index |
| `stock_data/sh/` × 2,400 | ~1.8 GB | OHLCV (price section + forecast TA) |
| `stock_data/sz/` × 3,000 | ~2.2 GB | same |
| `stock_data/fina_indicator/` × 5,200 | ~150 MB | quarterly fundamentals |
| `stock_data/static_features/*.csv` | ~10 MB | sectors, index members |
| `stock_data/stock_list.csv` + `stock_sectors.csv` | <1 MB | universe |
| `stock_data/news_sentiment_qwen.csv` | ~3 MB | sentiment trend |

Optional but useful:
| `stock_data/news_corpus_dedup_codes.parquet` | ~70 MB | regex-confirmed positives for the linker reranker |

**Total upload: ~5.5 GB** (vs ~13 GB if you also push the derived qa/
files).

```bash
HOST=region-1.autodl.com
PORT=22000
DEST=root@$HOST:/root/autodl-tmp/fortune/stock_data/
SSH="ssh -p $PORT"
rsync -avzP -e "$SSH" \
    stock_data/news_corpus_dedup.parquet \
    stock_data/news_corpus_dedup_codes.parquet \
    stock_data/sh/ stock_data/sz/ \
    stock_data/fina_indicator/ stock_data/static_features/ \
    stock_data/stock_list.csv stock_data/stock_sectors.csv \
    stock_data/news_sentiment_qwen.csv \
    $DEST
```

### Build sequence on the VM

In dependency order. Each command writes its output under
`stock_data/qa/`:

```bash
cd /root/autodl-tmp/fortune       # or wherever you cloned the repo
source venv_vllm/bin/activate

# 1. Build alias dict from raw CSVs (~5 sec)
python -m qa.build_alias_dict

# 2. Build the news linker output (~1 min on 1.95 M rows, CPU)
#    Reads: news_corpus_dedup.parquet + qa/aliases.json
#    Writes: qa/news_linked.parquet  ← THIS IS WHAT THE API NEEDS
python -m qa.linker.predict

# 3. Mine concept tags from the linker output (~30 sec)
python -m qa.build_concept_tags

# 4. Pull main_business / business_scope from Tushare (~30 sec)
#    Configure dl/config.py with TUSHARE_TOKEN first if not already.
python -m qa.fetch_company_business

# 5. Build the entity FAISS index (~30 sec on GPU; bge-m3 must be cached)
python -m qa.build_entity_index

# 6. (Optional) Train the linker reranker (~5 min)
python -m qa.linker.train_reranker

# 7. (Optional) Re-run linker with reranker boost (~30 min on 5090)
python -m qa.linker.predict --use_dense

# 8. Build the news FAISS index (~50 min on RTX 5090)
#    Reads: news_corpus_dedup.parquet
#    Writes: qa/news.faiss + qa/news_meta.parquet
python -m qa.build_news_index
```

After step 2 the API server can already start (see §10) — steps 3–8
add semantic news search and concept-tag fallback. Step 8 is the long
pole; skip it if you only want fundamentals + alias-driven queries
working today.

### Verify after build

```bash
ls -lh stock_data/qa/
# Expected:
#   aliases.json          ~3 MB
#   news_linked.parquet   ~134 MB
#   concept_tags.json     ~1 MB
#   company_business.csv  ~5 MB
#   entities.faiss        ~22 MB
#   entities.parquet      ~1 MB
#   news.faiss            ~8.5 GB    (after step 8)
#   news_meta.parquet     ~70 MB     (after step 8)
```

## 12. Run the full service + share the UI

Three screen sessions = three long-running processes. Make sure §1
env vars are loaded (`source ~/.bashrc`) and the venv is activated.

```bash
cd /root/autodl-tmp/fortune
source venv_vllm/bin/activate

# ─── Terminal A — vLLM (Qwen on GPU at :8000) ───
screen -S vllm
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --quantization awq_marlin \
    --port 8000
# Ctrl+A then D to detach

# ─── Terminal B — FastAPI (retrieval + cache + log at :8080) ───
screen -S api
python -m qa.api.server \
    --vllm-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --host 0.0.0.0 --port 8080
# Ctrl+A then D

# ─── Terminal C — Gradio UI (at :7860) ───
screen -S ui
python -m qa.api.gradio_app \
    --api http://127.0.0.1:8080 \
    --host 0.0.0.0 --port 7860
# Ctrl+A then D
```

Reattach any session with `screen -r <name>`. List with `screen -ls`.

Note `http://localhost:8000/v1` (with the `/v1` suffix) — that's the
OpenAI-compatible base URL. A typo like `/v` will load the API server
fine but every `/ask` call fails on the upstream.

### Verify locally on the VM

```bash
curl -s http://localhost:8080/healthz                # {"status":"ok"}
curl -s http://localhost:8080/vllm_health            # served_models lists Qwen
curl -s http://localhost:7860/ | head -3             # Gradio HTML
```

### Three ways to share the UI publicly

#### Option A — autodl built-in tunnel (recommended for autodl)

autodl's web console has a port-forwarding feature ("自定义服务"):

1. Instance dashboard → **更多 → 自定义服务**
2. Add port `7860` → autodl gives a URL like
   `https://u123456-port7860.region-1.autodl.com`
3. Send that URL. HTTPS, no expiry, autodl-native.

#### Option B — Gradio's built-in share tunnel (instant, 72 h)

Restart Terminal C with `--share`:

```bash
python -m qa.api.gradio_app \
    --api http://127.0.0.1:8080 \
    --host 0.0.0.0 --port 7860 --share
```

On Linux, Gradio auto-downloads `frpc` on first run (no Defender
issues like on Windows). Output shows `Running on public URL:
https://<random>.gradio.live`. Free, HTTPS, expires in 72 h.

#### Option C — Cloudflare Tunnel (no expiry, free)

```bash
# One-time install on the VM
wget -qO cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# Open a 4th screen session for the tunnel
screen -S tunnel
cloudflared tunnel --url http://localhost:7860
# prints: https://<random>.trycloudflare.com
# Ctrl+A then D
```

Stays alive while `cloudflared` runs. Stop with `screen -r tunnel`
then Ctrl+C. Free, HTTPS, no expiry, no account.

### Add basic auth before sharing

Anyone with the public URL can hit your GPU. Edit
`qa/api/gradio_app.py:demo.launch(...)` to add:

```python
demo.launch(server_name=args.host, server_port=args.port,
            share=args.share, auth=("guest", "your-shared-password"))
```

Restart Terminal C. Visitors will see a login screen first.

### Send users this cheat sheet

```
🔗 https://<your-public-url>

试试这些问题：
  • 600519.SH 业绩怎么样
  • 茅台最近有什么新闻
  • 锂电池龙头股有哪些
  • 钙钛矿电池标的
  • 宁德时代前景如何      ← 触发"模型预测"板块
  • 比亚迪未来走势怎样     ← 触发"模型预测"板块
  • 美联储加息对A股的影响  ← 政策类问题
```

The forecast keywords (`前景`, `未来走势`, `预测`, `看多`, `看空`,
`后市`) trigger the new model-prediction block (§Phase 2 forecast
integration) with the disclaimer "基于模型预测，仅供参考，不构成
投资建议".

## 13. Smoke-test curl commands

Quick copy-paste recipes for verifying the stack post-deployment.
All run on the VM (or anywhere the API is reachable on `:8080`).

### A. Health checks

```bash
# Our API
curl -s http://localhost:8080/healthz
# {"status":"ok"}

# vLLM is reachable + actually serving the expected model
curl -s http://localhost:8080/vllm_health
# {"status":"ok","vllm_url":"http://localhost:8000/v1",
#  "served_models":["Qwen/Qwen2.5-32B-Instruct-AWQ"],"cache_hits":0}
```

### B. Direct /ask — answer + meta

```bash
curl -s -X POST http://localhost:8080/ask \
     -H 'Content-Type: application/json' \
     --data '{"query":"600519.SH 业绩怎么样","top_k":5}' \
   | python -c "
import json, sys
d = json.load(sys.stdin)
print(f'ts_codes:    {d[\"ts_codes\"]}')
print(f'n_articles:  {d[\"n_articles\"]}')
print(f'context_chars: {d[\"context_chars\"]}')
print(f'elapsed_seconds: {d[\"elapsed_seconds\"]}')
print(f'cached:      {d.get(\"cached\")}')
print()
print(d['answer'])
"
```

### C. Forecast sanity test (the `X / Y` / `P%` check)

The system prompt deliberately uses `X / Y` / `P%` as placeholder
text. If those *literally* appear in the answer, the `模型预测`
block didn't reach Qwen — usually because `stock_data/modelfactory/`
isn't populated on the VM yet (see §11).

```bash
# Forecast-flavoured query → should cite real numbers
curl -s -X POST http://localhost:8080/ask \
     -H 'Content-Type: application/json' \
     --data '{"query":"000088.SZ 未来走势怎样","top_k":3}' \
   | python -c "
import json, sys, re
d = json.load(sys.stdin)
ans = d['answer']
print(f'ctx_chars: {d[\"context_chars\"]}')
print()
print(ans)
print('---')
if re.search(r'\bX\s*/\s*Y\b|\bP\s*%', ans):
    print('  ✗ FAIL — answer contains X/Y or P% placeholders.')
    print('     Check that stock_data/modelfactory/{live,runs}/ are populated')
    print('     on the VM, then restart the API server (vLLM stays up).')
else:
    print('  ✓ ok — no placeholder leakage')
"
```

A clean answer should mention concrete numbers like `14 / 30 看多`,
`53.6%`, `+0.32%`, plus an RSI / momentum trend, and end with the
disclaimer `基于模型预测，仅供参考，不构成投资建议`.

### D. Streaming /ask_stream (Server-Sent Events)

```bash
curl -N -X POST http://localhost:8080/ask_stream \
     -H 'Content-Type: application/json' \
     --data '{"query":"宁德时代前景如何","top_k":3}'
```

`-N` disables curl's output buffering so you see tokens land as they
stream. You'll see `event: meta`, then a long sequence of
`event: token`, then a final `event: done` with the full answer +
elapsed time.

### E. Aliases dropdown helper

```bash
# Quick name → ts_code lookup (used by Gradio's stock-filter dropdown)
curl -s "http://localhost:8080/aliases?prefix=茅台&limit=5"
# [{"ts_code":"600519.SH","name":"贵州茅台","symbol":"600519"}, ...]
```

### F. From your laptop (instead of from the VM)

If the API is ssh-port-forwarded or exposed via autodl 自定义服务,
swap `localhost:8080` for the public URL:

```bash
API=https://u123456-port8080.region-1.autodl.com   # your public URL
curl -s -X POST $API/ask \
     -H 'Content-Type: application/json' \
     --data '{"query":"光伏龙头股票","top_k":3}'
```
