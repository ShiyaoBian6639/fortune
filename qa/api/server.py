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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request


QA_LOG_PATH = Path('stock_data/qa/qa_log.jsonl')
_log_lock = threading.Lock()


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
                                       entity_meta='stock_data/qa/entities.parquet')
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
    max_context_tokens: int = 3000


@app.get('/healthz')
def healthz():
    return {'status': 'ok' if 'engine' in _state else 'starting'}


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
    t0 = time.time()
    out = _state['engine'].ask(
        body.query, _state['retriever'], _state['builder'],
        top_k=body.top_k, max_context_tokens=body.max_context_tokens,
    )
    out['elapsed_seconds'] = time.time() - t0

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
    return out


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
