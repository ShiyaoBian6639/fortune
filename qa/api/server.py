"""
FastAPI server for the stock-research Q&A engine (vLLM backend).

The heavy LLM lives in a separate vLLM process. This server is the
retrieval + context-builder + cache layer that talks to vLLM via the
OpenAI-compatible API.

Endpoints
---------
POST /ask
    body: {"query": str, "top_k": int = 5, "max_context_tokens": int = 3200}
    returns: {question, answer, ts_codes, n_articles, context_chars,
              elapsed_seconds, cached}

POST /ask_stream
    Same body, returns SSE stream:
        event: meta   data: {ts_codes, n_articles, context_chars}
        event: token  data: {text: '...'}                   (many)
        event: done   data: {elapsed_seconds, cached, full_answer}

GET  /healthz                       {"status": "ok"|"starting"}
GET  /aliases?prefix=...&limit=50   dropdown UI helper
GET  /vllm_health                   proxies vLLM /health for diagnosis

Run after vLLM is up on :8000:
    ./venv_vllm/bin/python -m qa.api.server \
        --vllm-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --port 8080
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from qa.rag.retriever import Retriever
from qa.rag.context_builder import ContextBuilder
from qa.rag.qa_engine import QAEngine
from qa.rag.forecast import ForecastProvider


QA_LOG_PATH = Path('stock_data/qa/qa_log.jsonl')
_log_lock = threading.Lock()


# ─── LRU response cache ────────────────────────────────────────────────────
# Identical query strings hit again return the stored answer instantly.
# vLLM has its own prefix cache so even un-cached repeats are fast, but
# this avoids the round-trip + re-tokenisation entirely.
_CACHE_SIZE = 256
_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(query: str, top_k: int) -> str:
    return f"{top_k}\x1f{query.strip()}"


def cache_get(query: str, top_k: int):
    k = _cache_key(query, top_k)
    with _cache_lock:
        if k in _cache:
            _cache.move_to_end(k)
            return _cache[k]
    return None


def cache_put(query: str, top_k: int, value: dict):
    k = _cache_key(query, top_k)
    with _cache_lock:
        _cache[k] = value
        _cache.move_to_end(k)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)


def _log_interaction(record: dict):
    """Append one JSON line to the Q&A log. Thread-safe append."""
    try:
        QA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _log_lock:
            with open(QA_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
    except Exception as e:
        print(f"[qa_log] write failed: {e}", flush=True)


_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] initialising QA stack...", flush=True)
    _state['retriever'] = Retriever('stock_data/qa/aliases.json',
                                       'stock_data/qa/news_linked.parquet',
                                       entity_index='stock_data/qa/entities.faiss',
                                       entity_meta='stock_data/qa/entities.parquet',
                                       news_index='stock_data/qa/news.faiss',
                                       news_meta='stock_data/qa/news_meta.parquet')
    # Pre-warm bge-m3. With 32 GB VRAM (5090) we keep it on GPU by
    # default — set QA_EMBED_DEVICE=cpu if you'd rather reserve VRAM.
    print("[api] pre-warming bge-m3 ...", flush=True)
    _t0 = time.time()
    _state['retriever']._ensure_embedder()
    print(f"[api] bge-m3 ready in {time.time()-_t0:.1f}s", flush=True)
    _state['builder'] = ContextBuilder('stock_data/qa/aliases.json')
    # ForecastProvider — rendered into context only when the query is
    # forecast-flavoured (前景 / 未来走势 / 预测 / ...). Loads ~5 MB of
    # live model CSVs at startup; query-time cost is microseconds.
    try:
        _state['forecast'] = ForecastProvider()
    except Exception as e:
        print(f"[api] forecast provider unavailable: {e}", flush=True)
        _state['forecast'] = None
    _state['engine']  = QAEngine(vllm_url=app.state.vllm_url,
                                    model=app.state.model)
    with open('stock_data/qa/aliases.json', 'r', encoding='utf-8') as f:
        _state['aliases'] = json.load(f)
    print(f"[api] ready (vllm={app.state.vllm_url} model={app.state.model})",
          flush=True)
    yield


app = FastAPI(title='A-Share Q&A', version='0.2.0-vllm', lifespan=lifespan)


class AskBody(BaseModel):
    query: str
    top_k: int = 5
    max_context_tokens: int = 3200


@app.get('/healthz')
def healthz():
    return {'status': 'ok' if 'engine' in _state else 'starting'}


@app.get('/vllm_health')
def vllm_health():
    """Proxy /health from the vLLM server. Useful for end-to-end
    sanity checks (does our process see vLLM, does vLLM respond)."""
    if 'engine' not in _state:
        return {'status': 'starting'}
    try:
        models = _state['engine'].client.models.list().data
        return {'status': 'ok',
                'vllm_url': _state['engine'].vllm_url,
                'served_models': [m.id for m in models],
                'cache_hits': len(_cache)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.get('/aliases')
def aliases(prefix: Optional[str] = None, limit: int = 50):
    out = []
    p = (prefix or '').strip().lower()
    for ts, v in _state['aliases'].items():
        n = v.get('name', '')
        if p:
            if not (p in ts.lower() or p in n.lower() or p in v.get('symbol', '')):
                continue
        out.append({'ts_code': ts, 'name': n, 'symbol': v.get('symbol', '')})
        if len(out) >= limit:
            break
    return out


@app.post('/ask')
def ask(body: AskBody, request: Request):
    if 'engine' not in _state:
        raise HTTPException(503, 'engine still loading')

    # LRU cache check
    cached = cache_get(body.query, body.top_k)
    if cached is not None:
        out = dict(cached)
        out['elapsed_seconds'] = 0.0
        out['cached'] = True
        return out

    t0 = time.time()
    out = _state['engine'].ask(
        body.query, _state['retriever'], _state['builder'],
        top_k=body.top_k, max_context_tokens=body.max_context_tokens,
        forecast=_state.get('forecast'),
    )
    out['elapsed_seconds'] = time.time() - t0
    out['cached'] = False

    _log_interaction({
        'timestamp':   _dt.datetime.now().isoformat(timespec='seconds'),
        'client':      request.client.host if request.client else None,
        'query':       body.query,
        'top_k':       body.top_k,
        'ts_codes':    out.get('ts_codes', []),
        'n_articles':  out.get('n_articles', 0),
        'context_chars': out.get('context_chars', 0),
        'elapsed_seconds': round(out['elapsed_seconds'], 2),
        'answer':      out.get('answer', ''),
    })
    cache_put(body.query, body.top_k, out)
    return out


@app.post('/ask_stream')
def ask_stream(body: AskBody, request: Request):
    """SSE endpoint — streams answer tokens as vLLM emits them.

    No GPU lock concerns: vLLM serialises GPU access internally and
    the OpenAI client iterator cleans up its HTTP connection on
    consumer disconnect.
    """
    if 'engine' not in _state:
        raise HTTPException(503, 'engine still loading')

    cached = cache_get(body.query, body.top_k)

    def iter_sse():
        # Cache hit → emit the cached answer as a single token.
        if cached is not None:
            meta_payload = {k: cached[k] for k in ('ts_codes','n_articles','context_chars')}
            yield f"event: meta\ndata: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"
            yield f"event: token\ndata: {json.dumps({'text': cached['answer']}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'elapsed_seconds': 0.0, 'cached': True, 'full_answer': cached['answer']}, ensure_ascii=False)}\n\n"
            return

        engine = _state['engine']
        retriever = _state['retriever']
        builder = _state['builder']
        meta = None
        full_text_parts: list[str] = []
        t0 = time.time()

        gen = engine.ask_stream(body.query, retriever, builder,
                                  top_k=body.top_k,
                                  max_context_tokens=body.max_context_tokens,
                                  forecast=_state.get('forecast'))
        try:
            for ev in gen:
                if ev['event'] == 'meta':
                    meta = ev
                    payload = {k: ev[k] for k in ('ts_codes','n_articles','context_chars')}
                    yield f"event: meta\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif ev['event'] == 'token':
                    full_text_parts.append(ev['text'])
                    yield f"event: token\ndata: {json.dumps({'text': ev['text']}, ensure_ascii=False)}\n\n"
                elif ev['event'] == 'done':
                    full = ''.join(full_text_parts)
                    elapsed = time.time() - t0
                    yield f"event: done\ndata: {json.dumps({'elapsed_seconds': elapsed, 'cached': False, 'full_answer': full}, ensure_ascii=False)}\n\n"
                    if meta is not None:
                        rec = {
                            'timestamp':   _dt.datetime.now().isoformat(timespec='seconds'),
                            'client':      request.client.host if request.client else None,
                            'query':       body.query,
                            'top_k':       body.top_k,
                            'ts_codes':    meta['ts_codes'],
                            'n_articles':  meta['n_articles'],
                            'context_chars': meta['context_chars'],
                            'elapsed_seconds': round(elapsed, 2),
                            'answer':      full,
                        }
                        _log_interaction(rec)
                        cache_put(body.query, body.top_k, {
                            'question':      body.query,
                            'answer':        full,
                            'ts_codes':      meta['ts_codes'],
                            'n_articles':    meta['n_articles'],
                            'context_chars': meta['context_chars'],
                        })
        finally:
            try:
                gen.close()
            except Exception:
                pass

    return StreamingResponse(iter_sse(), media_type='text/event-stream')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--vllm-url', default='http://localhost:8000/v1',
                   help='vLLM OpenAI-compatible base URL')
    p.add_argument('--model', default='Qwen/Qwen2.5-32B-Instruct-AWQ',
                   help='Served model name (must match vllm serve <model>)')
    args = p.parse_args()

    import uvicorn
    app.state.vllm_url = args.vllm_url
    app.state.model    = args.model
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
