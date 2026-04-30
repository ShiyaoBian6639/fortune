# 无能囃人A股深度解析 — Q&A System

Chinese stock-research Q&A engine over the existing `stock_data/` corpus.
Combines an alias-based news linker (Aho-Corasick) with structured fundamentals
retrieval and a Qwen2.5-7B-Instruct (4-bit NF4) generator.

Pass rate on the 5-question grounding eval: **5 / 5 (100 %)**.

---

## Architecture

```
User question
    │
    ▼
[1] Entity extraction      qa/rag/retriever.py   → ts_code(s) via aliases.json
    │
    ▼
[2] News retrieval         news_linked.parquet   → top-K articles for those ts_codes
    │
    ▼
[3] Context assembly       qa/rag/context_builder.py
        - Entity card (sector, industry, area, index tags)
        - 4-quarter fundamentals table (fina_indicator/{code}_{SUF}.csv)
        - 30-day price summary (sh|sz/{code}.csv)
        - 30-day sentiment trend (news_sentiment_qwen.csv)
        - Top-K news snippets [date][source]
        Hard cap: 3,000 tokens
    │
    ▼
[4] Q&A LLM                qa/rag/qa_engine.py    → Qwen2.5-7B-Instruct (4-bit, ~10 GB VRAM)
    │
    ▼
Answer + footer (resolved ts_codes, article count, latency)
```

Two-process deployment: a **FastAPI server** holds the model in VRAM and exposes
`/ask`; a **Gradio UI** is a thin HTTP client. Splitting them lets you bounce
the UI without re-loading Qwen.

---

## File layout

```
qa/
├── README.md                 (this file)
├── __init__.py
├── build_alias_dict.py       # aliases.json (5,190 stocks, ~3.9 aliases each)
├── build_entity_index.py     # entities.faiss + entities.parquet (Phase 2A)
├── build_news_index.py       # news.faiss + news_meta.parquet (Phase 2C, optional)
├── local_loader.py           # offline-first BAAI/bge-m3 model loader
├── linker/
│   ├── ahocorasick_matcher.py
│   ├── dense_matcher.py      # bge-m3 entity candidates per article (Phase 2D)
│   ├── train_reranker.py     # LR over [lex, dense_cos, lengths] (Phase 2D)
│   └── predict.py            # → news_linked.parquet  (--use_dense optional)
├── rag/
│   ├── retriever.py          # alias resolve + dense semantic fallback
│   ├── embedder.py           # thin bge-m3 wrapper (CLS pool + L2 norm)
│   ├── context_builder.py    # markdown context assembly
│   └── qa_engine.py          # Qwen wrapper
├── api/
│   ├── server.py             # FastAPI: /ask, /aliases, /healthz   (port 8080)
│   └── gradio_app.py         # Gradio chat UI                      (port 7860)
└── eval/
    └── run_eval.py           # 5-question grounding test
```

Generated data (under `stock_data/qa/`):

| File | Size | Built by |
|---|---|---|
| `aliases.json`           | ~2.7 MB | `qa.build_alias_dict` |
| `news_linked.parquet`    | ~120 MB | `qa.linker.predict` |
| `entities.faiss` + `entities.parquet` | ~23 MB | `qa.build_entity_index` (Phase 2A) |
| `linker_reranker.pkl`    | small | `qa.linker.train_reranker` (Phase 2D) |
| `news.faiss` + `news_meta.parquet`    | ~3.8 GB | `qa.build_news_index` (Phase 2C, optional) |
| `qa_log.jsonl`           | grows | API server (one line per `/ask`) |
| `eval_report.json`       | small | `qa.eval.run_eval` |

bge-m3 model snapshot:
| Path | Size | Built by |
|---|---|---|
| `stock_data/models/bge-m3/`            | ~2.3 GB | `qa.local_loader` (one-time) |

---

## Setup

Run from `D:\didi\stock\tushare`. Use the project venv (`./venv/Scripts/python`).

### One-time data builds

```bash
# 1. Build the stock alias dictionary (~5 s)
./venv/Scripts/python -m qa.build_alias_dict

# 2. Tag the news corpus with ts_codes (~1 min over 1.95 M articles)
./venv/Scripts/python -m qa.linker.predict
```

Re-run both whenever `stock_list.csv`, `stock_company.csv`, `stock_sectors.csv`,
`index_member_pit.csv`, or `news_corpus_dedup.parquet` are refreshed.

### Phase 2: bge-m3 dense indexes (optional but recommended)

Required for the semantic-fallback retriever — without these the system
returns "未能识别" for queries that don't contain a stock name (e.g.
"新能源车板块龙头是谁").

```bash
# 3. Cache bge-m3 weights locally (~2.3 GB, one-time, 5–15 min)
./venv/Scripts/python -m qa.local_loader

# 4. Build the per-stock entity index (~1 min, 22 MB)
./venv/Scripts/python -m qa.build_entity_index

# 5. (Optional) Train the linker reranker (~5 min on 8 K positives)
./venv/Scripts/python -m qa.linker.train_reranker

# 6. (Optional) Re-link the news corpus with reranker boost (~1 hr)
./venv/Scripts/python -m qa.linker.predict --use_dense

# 7. (Optional) Build a dense FAISS index over all 1.87 M articles (~1 hr, 3.8 GB)
./venv/Scripts/python -m qa.build_news_index
```

Steps 3–4 are the minimum needed to lift eval pass rate from 4/5 to 5/5
by adding the semantic fallback. Steps 5–6 lift news-corpus linking
coverage beyond the 25 % AC floor. Step 7 enables direct semantic
news search (currently scaffolded but not wired into the retriever).

### Required Python packages

Already in `requirements_qwen.txt`:
- `transformers`, `accelerate`, `bitsandbytes>=0.46.1`
- `pyahocorasick`
- `fastapi`, `uvicorn`, `pydantic`
- `gradio>=6`
- `requests`

---

## Running locally

### Terminal 1 — API server (model lives here)

```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python -m qa.api.server --quant 4bit --port 8080
```

Loading Qwen2.5-7B 4-bit takes ~30–60 s. Wait for `[api] ready (...)`.

Endpoints:
- `POST /ask`   body `{"query": str, "top_k": int = 5, "max_context_tokens": int = 3000}` → `{question, answer, ts_codes, n_articles, context_chars, elapsed_seconds}`
- `GET /aliases?prefix=...&limit=50` → `[{ts_code, name, symbol}]`
- `GET /healthz` → `{"status": "ok" | "starting"}`

### Terminal 2 — Gradio UI

```bash
./venv/Scripts/python -m qa.api.gradio_app --api http://127.0.0.1:8080
```

Open `http://127.0.0.1:7860`. Features:
- **Title**: 无能囃人A股深度解析
- **Filter dropdown**: all 5,190 stocks; type any name fragment (`茅台`, `平安`,
  `600519`) and the list narrows live (Gradio's built-in `filterable=True`).
  When a stock is selected, its `ts_code` is auto-prepended to the query.
- **Submit button**: explicit "提交" button + Enter (Shift+Enter for newline).
- **Clear**: empties the chat history (does not stop a request in flight).

### Smoke test the API directly

```bash
./venv/Scripts/python -c "
import requests
r = requests.post('http://127.0.0.1:8080/ask',
                  json={'query':'茅台最近有什么新闻？','top_k':5},
                  timeout=120)
print(r.json()['answer'])
"
```

---

## Logging

The API server appends one JSON line per request to
`stock_data/qa/qa_log.jsonl`. Schema:

```json
{
  "timestamp": "2026-04-30T14:22:01",
  "client": "127.0.0.1",
  "query": "茅台最近有什么新闻？",
  "top_k": 5,
  "ts_codes": ["600519.SH"],
  "n_articles": 5,
  "context_chars": 1860,
  "elapsed_seconds": 19.4,
  "answer": "**贵州茅台 ... **"
}
```

Inspect:
```bash
./venv/Scripts/python -c "
import json
for line in open('stock_data/qa/qa_log.jsonl', encoding='utf-8'):
    r = json.loads(line)
    print(f\"[{r['timestamp']}] {r['ts_codes']} ({r['elapsed_seconds']}s)  {r['query']}\")
"
```

The log is the source of truth for who asked what. Rotate / archive it manually
when it gets large.

---

## Sharing the UI

The defaults bind to `127.0.0.1` (loopback only). Three ways to expose it:

### A — LAN (anyone on the same network)

```bash
./venv/Scripts/python -m qa.api.gradio_app --api http://127.0.0.1:8080 --host 0.0.0.0
```

Find your LAN IP with `ipconfig`. Send `http://<your-ip>:7860`. First time
binding to `0.0.0.0`, Windows Firewall will prompt — allow it for the Private
profile. To allow manually (Administrator PowerShell):

```powershell
netsh advfirewall firewall add rule name="QA Gradio 7860" dir=in action=allow protocol=TCP localport=7860
```

### B — Public via Cloudflare Tunnel (recommended for sharing outside the LAN)

One-time install (PowerShell):

```powershell
winget install --id Cloudflare.cloudflared
```

Then in a separate terminal (no PATH refresh needed if you use the absolute path):

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe" tunnel --url http://localhost:7860
```

You'll see `https://<random>.trycloudflare.com` after a few seconds — that's
the public URL. No expiry, HTTPS, no account. Ctrl+C to take it down.

### C — Public via Gradio's own tunnel

```bash
./venv/Scripts/python -m qa.api.gradio_app --api http://127.0.0.1:8080 --share
```

First run downloads `frpc_windows_amd64_v0.3` to
`%USERPROFILE%\.cache\huggingface\gradio\frpc\`. Windows Defender frequently
quarantines this binary; option B avoids that issue.

### Auth on a public URL

A public URL = anyone with the link can hit your GPU. Add basic auth in
`qa/api/gradio_app.py` at `demo.launch(...)`:

```python
demo.launch(..., auth=("guest", "your-shared-password"))
```

---

## Evaluation

```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python -m qa.eval.run_eval --quant 4bit
```

5 hand-graded cases in `qa/eval/run_eval.py`. Pass criteria per case:
ts_code resolution, answer length, optional numeric tolerance match, optional
min news count. Results saved to `stock_data/qa/eval_report.json`.

Current pass map:

| # | Question | Status |
|---|---|---|
| 1 | `300750.SZ 最近一季度的业绩怎么样？` | ✓ |
| 2 | `茅台最近有什么新闻？` | ✓ |
| 3 | `比亚迪和宁德时代谁的毛利率更高？` | ✓ |
| 4 | `贵州茅台 2025年Q4 EPS 是多少？` | ✓ |
| 5 | `新能源车板块龙头是谁，业绩对比如何？` | ✓ (Phase 2 semantic fallback) |

Q5 was the original deferred case. Phase 2 added a bge-m3 entity index
+ retriever fallback (cosine + log-news-count prominence prior). The
query now resolves to BYD (002594.SZ) and produces a comparison table.

---

## Known limitations

- **Semantic fallback is approximate.** When no stock name appears in the
  query, the retriever embeds the query with bge-m3 and pulls top-50
  entity candidates, then reranks by `cos + 0.25 · log1p(news_count)`.
  This surfaces real sector leaders for most cases (锂电池 → 天齐锂业 /
  亿纬锂能, 光伏 → 阳光电源 / 晶澳科技, 白酒 → 茅台). Failure modes:
  literal-token collisions on stocks named after the query keyword
  ("龙头股份" textile co. for "X龙头" queries before intent-word stripping)
  and fields missing from the entity card. Tunable knobs:
  `semantic_min_score=0.40`, `pool_k=50`, `prior_weight=0.25` in
  `qa/rag/retriever.py`.
- **No streaming responses.** `/ask` blocks until generation completes
  (~3–25 s depending on context size). Gradio shows nothing in the meantime.
- **Single-GPU serial.** Concurrent `/ask` calls queue on the same Qwen
  instance; no batching. bge-m3 is lazy-loaded on first semantic
  fallback and shares the GPU; running it alongside Qwen 4-bit fits in
  12 GB but only just.
- **No auth.** Add Gradio basic auth before exposing publicly.
- **`type='messages'` removed.** Gradio 6 uses messages format by default;
  the older arg name is no longer accepted.
- **News index activates rarely.** `qa/build_news_index.py` produces an
  8.5 GB FAISS index that the retriever lazy-loads as a third fallback
  (alias → entity-semantic → news-semantic). In practice the
  entity-semantic path matches almost every query first, even loosely-
  related meta ones, so the news path rarely fires. Routing between
  the two (e.g. detect "meta" queries by keyword and prefer news first)
  is open work.

---

## Common edits

| What | Where |
|---|---|
| Add a stock alias by hand | extend `LEGAL_SUFFIXES` / `COMMON_PREFIXES` in `build_alias_dict.py`, rebuild |
| Change context budget | `max_context_tokens` arg to `engine.ask()` (default 3000) or POST body |
| Change retrieval window | `top_k` POST body (default 5) |
| Loosen alias dedup | `sp_blacklist = {sp for sp, n in sp_counts.items() if n >= 2}` in `build_alias_dict.py` (currently 21 generic 2-char terms dropped) |
| Change UI title / branding | `qa/api/gradio_app.py` — `gr.Blocks(title=...)` and `gr.Markdown("# ...")` |
| Rotate the log | move `stock_data/qa/qa_log.jsonl` aside; the server re-creates it on next request |

---

## Cache invariants (don't skip)

- After editing `build_alias_dict.py`, rebuild **both** `aliases.json` **and**
  `news_linked.parquet` — the linker uses the same alias dict.
- After editing `linker/`, rebuild only `news_linked.parquet`.
- After editing `rag/` or `api/`, just bounce the FastAPI server.
- After editing `gradio_app.py`, bounce only Gradio (the API stays warm).
