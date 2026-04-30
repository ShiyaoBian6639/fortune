"""
FastAPI server for the stock-research Q&A engine.

Endpoints
---------
POST /ask
    body: {"query": str, "top_k": int = 5}
    returns: {
        "question":    str,
        "answer":      str,
        "ts_codes":    list[str],
        "n_articles":  int,
        "context_chars": int,
        "elapsed_seconds": float,
    }

GET /healthz   {"status": "ok"}

GET /aliases?prefix=...   list of {ts_code, name} for the dropdown UI.

Run:
    ./venv/Scripts/python -m qa.api.server --quant 4bit --port 8080
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
from typing import List, Optional

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


QA_LOG_PATH = Path('stock_data/qa/qa_log.jsonl')
_log_lock = threading.Lock()


# ─── LRU response cache ────────────────────────────────────────────────────
# Identical query strings hit again within the cache window return the
# stored answer instantly (full ~20 s saved). Keyed on the normalised
# query + top_k so different top_k values don't collide.
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
        # Never let logging take down a request
        print(f"[qa_log] write failed: {e}", flush=True)

from qa.rag.retriever import Retriever
from qa.rag.context_builder import ContextBuilder
from qa.rag.qa_engine import QAEngine


_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] initialising QA engine...", flush=True)
    _state['retriever'] = Retriever('stock_data/qa/aliases.json',
                                       'stock_data/qa/news_linked.parquet',
                                       entity_index='stock_data/qa/entities.faiss',
                                       entity_meta='stock_data/qa/entities.parquet',
                                       news_index='stock_data/qa/news.faiss',
                                       news_meta='stock_data/qa/news_meta.parquet')
    # Pre-warm bge-m3 (CPU by default) so the first semantic-fallback
    # query doesn't pay the ~20 s model-load tax. Defaults to CPU to
    # keep the 4070 Super's 12 GB VRAM clear for Qwen — set
    # QA_EMBED_DEVICE=cuda only if you have ≥18 GB VRAM.
    print("[api] pre-warming bge-m3 ...", flush=True)
    _t0 = time.time()
    _state['retriever']._ensure_embedder()
    print(f"[api] bge-m3 ready in {time.time()-_t0:.1f}s", flush=True)
    _state['builder'] = ContextBuilder('stock_data/qa/aliases.json')
    _state['engine']  = QAEngine(model_id=app.state.model_id,
                                    quant=app.state.quant)
    # Pre-load alias name list for the dropdown
    with open('stock_data/qa/aliases.json', 'r', encoding='utf-8') as f:
        _state['aliases'] = json.load(f)
    print(f"[api] ready (model={app.state.model_id} quant={app.state.quant})", flush=True)
    yield


app = FastAPI(title='A-Share Q&A', version='0.1.0', lifespan=lifespan)


class AskBody(BaseModel):
    query: str
    top_k: int = 5
    max_context_tokens: int = 3200


@app.get('/healthz')
def healthz():
    return {'status': 'ok' if 'engine' in _state else 'starting'}


@app.get('/gpu_stats')
def gpu_stats():
    """Live VRAM snapshot. ``spilled`` is True if peak allocation has
    ever exceeded physical VRAM — once that happens this run, CUDA
    has fallen back to shared/system memory and inference is 10–100×
    slower until the process restarts.
    """
    import torch
    if not torch.cuda.is_available():
        return {'cuda': False}
    total = torch.cuda.get_device_properties(0).total_memory
    peak = torch.cuda.max_memory_allocated()
    return {
        'cuda':          True,
        'device_name':   torch.cuda.get_device_name(0),
        'total_gb':      round(total / 1e9, 2),
        'allocated_gb':  round(torch.cuda.memory_allocated() / 1e9, 2),
        'reserved_gb':   round(torch.cuda.memory_reserved()  / 1e9, 2),
        'max_alloc_gb':  round(peak / 1e9, 2),
        'spilled':       peak > total * 0.97,
        'cache_hits':    len(_cache),
    }


@app.post('/gpu_reset_peak')
def gpu_reset_peak():
    """Clear torch.cuda.max_memory_allocated() so /gpu_stats peak
    starts fresh — useful when verifying a fix without restarting."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return {'ok': True}


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
    """SSE endpoint that streams answer tokens as they're generated.
    Drops perceived latency from ~15 s to <1 s for first tokens.

    Emits Server-Sent Events:
        event: meta   data: {ts_codes, n_articles, context_chars}
        event: token  data: {text: '...'}                   (many)
        event: done   data: {elapsed_seconds, full_answer}
    """
    if 'engine' not in _state:
        raise HTTPException(503, 'engine still loading')

    # Cache hit → emit the cached answer as a single token, then done.
    cached = cache_get(body.query, body.top_k)

    def iter_sse():
        nonlocal cached
        if cached is not None:
            yield f"event: meta\ndata: {json.dumps({k: cached[k] for k in ('ts_codes','n_articles','context_chars')}, ensure_ascii=False)}\n\n"
            yield f"event: token\ndata: {json.dumps({'text': cached['answer']}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'elapsed_seconds': 0.0, 'cached': True, 'full_answer': cached['answer']}, ensure_ascii=False)}\n\n"
            return

        engine = _state['engine']
        retriever = _state['retriever']
        builder = _state['builder']
        meta = None
        full_text_parts: list[str] = []
        t0 = time.time()
        # Wrap the generator in a try/finally so a client disconnect
        # forces ``gen.close()`` → triggers the engine generator's finally
        # → releases the GPU lock. Without this the lock leaks on early
        # disconnect and every subsequent /ask hangs forever.
        gen = engine.ask_stream(body.query, retriever, builder,
                                  top_k=body.top_k,
                                  max_context_tokens=body.max_context_tokens)
        try:
          for ev in gen:
            if ev['event'] == 'meta':
                meta = ev
                yield f"event: meta\ndata: {json.dumps({k: ev[k] for k in ('ts_codes','n_articles','context_chars')}, ensure_ascii=False)}\n\n"
            elif ev['event'] == 'token':
                full_text_parts.append(ev['text'])
                yield f"event: token\ndata: {json.dumps({'text': ev['text']}, ensure_ascii=False)}\n\n"
            elif ev['event'] == 'done':
                full = ''.join(full_text_parts)
                elapsed = time.time() - t0
                yield f"event: done\ndata: {json.dumps({'elapsed_seconds': elapsed, 'cached': False, 'full_answer': full}, ensure_ascii=False)}\n\n"
                # Persist to log + cache after streaming completes.
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
            # Force the engine generator to finalize even if the SSE
            # consumer disconnected mid-stream — this runs the
            # generate_stream finally block which releases the GPU lock.
            try:
                gen.close()
            except Exception:
                pass

    return StreamingResponse(iter_sse(), media_type='text/event-stream')


@app.post('/lock_status')
def lock_status():
    """Diagnose lock state. ``locked`` is True if a generation is
    currently in flight; if it stays True with no active client,
    something leaked — POST /lock_force_release as a last resort."""
    if 'engine' not in _state:
        return {'locked': False, 'note': 'engine not loaded'}
    lk = _state['engine']._gpu_lock
    held = not lk.acquire(blocking=False)
    if not held:
        lk.release()
    return {'locked': held}


@app.post('/lock_force_release')
def lock_force_release():
    """Last-resort emergency: forcibly release the GPU lock without
    waiting for the generator to finalize. Only use when /lock_status
    shows locked=true and no request is actually in flight (e.g. after
    a browser disconnect that orphaned the generator). Restart the
    server if generation still hangs after this."""
    if 'engine' not in _state:
        return {'released': False, 'note': 'engine not loaded'}
    lk = _state['engine']._gpu_lock
    try:
        lk.release()
        return {'released': True}
    except RuntimeError:
        return {'released': False, 'note': 'lock was not held'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--quant', choices=['none', '8bit', '4bit'], default='4bit')
    args = p.parse_args()

    import uvicorn
    app.state.model_id = args.model
    app.state.quant    = args.quant
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
