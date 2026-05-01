"""
QA generation engine — thin client over a vLLM OpenAI-compatible server.

Previously this module loaded Qwen in-process via transformers + bnb-4bit
on a 12 GB 4070 Super. That setup hit several walls (KV-cache OOM,
generator-with-lock leaks, single-request serialisation, ~30 tok/s).

The new layout puts the LLM in a separate vLLM process owning the GPU
and exposes an OpenAI-compatible endpoint. This module just sends
chat-completion requests, optionally streamed. vLLM handles batching,
KV cache paging, concurrency, and AWQ-quantised serving — typically
3–5× faster end-to-end than the old setup, and supports many concurrent
users.

Architecture:

    [Gradio / curl / eval]            (HTTP)
            │
            ▼
    [qa.api.server  :8080]            (this repo's FastAPI)
            │
            ├── retriever  (alias / entity FAISS / news FAISS)
            ├── builder    (markdown context)
            └── QAEngine  ──(OpenAI client)──► vllm serve  :8000  (GPU)

Run vLLM separately:

    vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --quantization awq_marlin \
        --port 8000

Then start this repo's API server (``qa.api.server --vllm-url http://...``).
"""
from __future__ import annotations

import argparse
import re
import time
from typing import Iterator, Optional

from openai import OpenAI

from qa.rag.retriever import Retriever
from qa.rag.context_builder import ContextBuilder


# ─── Dynamic max-token heuristic ─────────────────────────────────────────────
_RE_SHORT  = re.compile(r'(是多少|多少元|多少钱|多少倍|什么时候|哪一年|哪天|是哪个|是谁|代码是|属于哪)')
_RE_LONG   = re.compile(r'(新闻|消息|动态|进展|影响|对比|比较|分析|趋势|为什么|怎么看|策略|展望|预测)')


def pick_max_new_tokens(question: str, default: int = 400) -> int:
    if _RE_SHORT.search(question): return 200
    if _RE_LONG.search(question):  return 700
    return default


# ─── System prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一名严谨的中国A股研究助手。

**核心规则：**
1. **只使用「资料」中的事实**。资料里没有就明确说"资料中未提供"。
2. **数字与股票代码必须严格摘自资料**——绝不凭空生成 EPS、ROE、收盘价、涨跌幅、ts_code 等数据。引用 ts_code 时必须照抄资料中给出的 6 位数字 + 市场后缀。
3. **引用新闻时**写明日期、来源（[2026-04-28] [eastmoney] 标题），不要发明新闻条目。
4. **不要在回答中出现"问题："、"请列出"等再次提问的语句。** 只回答用户当前的问题，不要继续问问题，不要列出后续问题。
5. **同一公司或短语不要在一段话里反复出现 3 次以上。** 一次清晰地说完。
6. 回答格式：
   - 业绩问题 → 用一段话总结趋势 + 关键季度指标。
   - 新闻问题 → 3-5 条最相关的新闻，编号从 1 开始连续。
   - 比较问题 → Markdown 表格直接对比。
   - 行业/概念查询 → 先点名 2-3 家代表公司，再给资料中的关键数据/新闻支持。
7. 用中文回答，专业但易读，控制在 600 字以内。"""


# ─── Engine ──────────────────────────────────────────────────────────────────
class QAEngine:
    """OpenAI-client wrapper around a running vLLM server.

    All the in-process Qwen/transformers complexity (threading lock,
    KV-cache management, attention backend selection, bnb quantisation)
    is gone — vLLM owns the GPU and we just send HTTP requests.
    """

    def __init__(self,
                 vllm_url: str = 'http://localhost:8000/v1',
                 model:    str = 'Qwen/Qwen2.5-32B-Instruct-AWQ',
                 max_new_tokens: int = 600,
                 temperature: float = 0.3,
                 timeout: float = 180.0,
                 # Kept for API compat with old call sites (ignored)
                 device: str = 'cuda',
                 quant: str | None = None,
                 model_id: str | None = None):
        self.vllm_url = vllm_url
        self.model    = model_id or model
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.timeout        = timeout
        self.client = OpenAI(base_url=vllm_url, api_key='EMPTY',
                              timeout=timeout)
        # Probe vLLM to fail fast if the server isn't reachable; pick the
        # served model name automatically when it differs from the default.
        try:
            served = self.client.models.list().data
            served_ids = [m.id for m in served]
            if served_ids and self.model not in served_ids:
                # vLLM's served name uses the same string as --model. If
                # the user passed a different ID, prefer the actual served
                # one to avoid 404s.
                print(f"[qa] requested model {self.model!r} not in "
                      f"{served_ids} — falling back to {served_ids[0]!r}")
                self.model = served_ids[0]
            print(f"[qa] vllm @ {vllm_url}  model={self.model}")
        except Exception as e:
            print(f"[qa] WARNING: vllm not reachable at {vllm_url} — {e}",
                  flush=True)

    # ─── Prompt assembly ──────────────────────────────────────────────────
    def _build_messages(self, question: str, context: str) -> list:
        # Few-shot example was leaking its "问题：宁德时代..." trailer
        # into outputs (Qwen would echo "问题：请列出X" at the end of its
        # response). System prompt + structured context is enough.
        return [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': f"资料：\n{context}\n\n问题：{question}"},
        ]

    def _extra(self) -> dict:
        # vLLM accepts these as extra fields — repetition_penalty +
        # no_repeat_ngram_size are forwarded to the underlying generate.
        return {
            'repetition_penalty': 1.18,
            'top_p': 0.92,
        }

    # ─── Generation ───────────────────────────────────────────────────────
    def generate(self, question: str, context: str,
                  max_new_tokens: int | None = None) -> str:
        max_new = max_new_tokens or self.max_new_tokens
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(question, context),
            max_tokens=max_new,
            temperature=self.temperature,
            extra_body=self._extra(),
        )
        return (resp.choices[0].message.content or '').strip()

    def generate_stream(self, question: str, context: str,
                         max_new_tokens: int | None = None) -> Iterator[str]:
        """Yield text deltas as the vLLM server emits them.

        No threading lock, no streamer queue, no thread.join. The OpenAI
        client iterator handles SSE under the hood; if the consumer
        disconnects the iterator simply ends and the underlying HTTP
        connection closes — vLLM cancels the request server-side.
        """
        max_new = max_new_tokens or self.max_new_tokens
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(question, context),
            max_tokens=max_new,
            temperature=self.temperature,
            extra_body=self._extra(),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ─── Retrieval + answer ───────────────────────────────────────────────
    def _retrieve_and_build(self, question, retriever, builder,
                              top_k, max_context_tokens):
        out = retriever.search(question, top_k=top_k)
        context = builder.build_for(out['ts_codes'], out['articles'],
                                      max_tokens=max_context_tokens)
        return out, context

    def ask(self, question: str,
             retriever: Retriever,
             builder: ContextBuilder,
             top_k: int = 5,
             max_context_tokens: int = 3200) -> dict:
        out, context = self._retrieve_and_build(question, retriever, builder,
                                                  top_k, max_context_tokens)
        if not out['ts_codes'] and not out['articles']:
            answer = ("未能识别出问题相关的A股或新闻。请尝试包含股票代码、"
                       "公司名称或具体行业关键词。")
        else:
            t0 = time.time()
            answer = self.generate(question, context,
                                     max_new_tokens=pick_max_new_tokens(question))
            print(f"[qa] generated in {time.time()-t0:.1f}s", flush=True)
        return {
            'question':      question,
            'answer':        answer,
            'ts_codes':      out['ts_codes'],
            'n_articles':    len(out['articles']),
            'context_chars': len(context),
        }

    def ask_stream(self, question: str,
                    retriever: Retriever,
                    builder: ContextBuilder,
                    top_k: int = 5,
                    max_context_tokens: int = 3200) -> Iterator[dict]:
        """Yield events:
            {'event':'meta',  'ts_codes':..., 'n_articles':..., 'context_chars':...}
            {'event':'token', 'text': '...'}            (zero or more)
            {'event':'done',  'elapsed_seconds': ...}
        """
        out, context = self._retrieve_and_build(question, retriever, builder,
                                                  top_k, max_context_tokens)
        yield {
            'event': 'meta',
            'ts_codes':      out['ts_codes'],
            'n_articles':    len(out['articles']),
            'context_chars': len(context),
        }
        if not out['ts_codes'] and not out['articles']:
            yield {'event': 'token',
                   'text': "未能识别出问题相关的A股或新闻。请尝试包含股票代码、"
                            "公司名称或具体行业关键词。"}
            yield {'event': 'done', 'elapsed_seconds': 0.0}
            return
        t0 = time.time()
        for piece in self.generate_stream(question, context,
                                            max_new_tokens=pick_max_new_tokens(question)):
            yield {'event': 'token', 'text': piece}
        yield {'event': 'done', 'elapsed_seconds': time.time() - t0}


# ─── CLI smoke test ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vllm-url', default='http://localhost:8000/v1')
    p.add_argument('--model',    default='Qwen/Qwen2.5-32B-Instruct-AWQ')
    p.add_argument('--questions', nargs='+', default=[
        '300750.SZ最近一季度的业绩怎么样？',
        '茅台最近有什么新闻？',
        '比亚迪和宁德时代谁的毛利率更高？',
    ])
    args = p.parse_args()

    r  = Retriever('stock_data/qa/aliases.json',
                    'stock_data/qa/news_linked.parquet',
                    entity_index='stock_data/qa/entities.faiss',
                    entity_meta='stock_data/qa/entities.parquet',
                    news_index='stock_data/qa/news.faiss',
                    news_meta='stock_data/qa/news_meta.parquet')
    cb = ContextBuilder('stock_data/qa/aliases.json')
    qa = QAEngine(vllm_url=args.vllm_url, model=args.model)

    for q in args.questions:
        print()
        print('=' * 70)
        print(f'Q: {q}')
        print('=' * 70)
        t0 = time.time()
        out = qa.ask(q, r, cb)
        print(f"resolved ts_codes: {out['ts_codes']}  "
              f"({out['n_articles']} articles, {out['context_chars']} ctx chars)  "
              f"{time.time()-t0:.1f}s")
        print()
        print(out['answer'])


if __name__ == '__main__':
    main()
