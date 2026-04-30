"""
Qwen-based Q&A engine — wraps Qwen2.5-Instruct for the stock-research task.

Loads the same Qwen2.5-7B-Instruct that `xgbmodel/qwen_sentiment.py` uses
(weights are HF-cached so no second download), but configured for free-form
generation instead of single-token sentiment classification.

Usage:
    qa = QAEngine(model_id='Qwen/Qwen2.5-7B-Instruct', quant='4bit')
    answer = qa.ask("贵州茅台最近一季度业绩如何", retriever, builder)

The retriever + context_builder produce structured Markdown context. We
prepend a strict system prompt that requires the model to ONLY use the
provided facts, never invent numbers, and cite the news date+source when
referencing news.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from threading import Thread
from typing import Iterator, List, Optional

import numpy as np

from qa.rag.retriever import Retriever
from qa.rag.context_builder import ContextBuilder


# Heuristic for picking max_new_tokens. Big chunks of tokens are the
# bottleneck (~30 tok/s on RTX 4070 Super at int4); not generating
# tokens we don't need is the cheapest real win.
_RE_SHORT  = re.compile(r'(是多少|多少元|多少钱|多少倍|什么时候|哪一年|哪天|是哪个|是谁|代码是|属于哪)')
_RE_LONG   = re.compile(r'(新闻|消息|动态|进展|影响|对比|比较|分析|趋势|为什么|怎么看|策略|展望|预测)')

def pick_max_new_tokens(question: str, default: int = 400) -> int:
    if _RE_SHORT.search(question): return 200
    if _RE_LONG.search(question):  return 700
    return default


SYSTEM_PROMPT = """你是一名严谨的中国A股研究助手。回答问题时必须遵守以下规则：

1. **只使用「资料」中的事实**。如果资料中没有该信息，必须明确说"资料中未提供"。
2. **数字必须严格摘自资料**——绝不凭空生成 EPS、ROE、收盘价、涨跌幅等数据。
3. **引用新闻时**写明日期、来源（[2026-04-28] [eastmoney] 标题），不要发明新闻条目。
4. 回答简洁清晰：
   - 业绩问题 → 用一段话总结趋势 + 列出关键季度指标。
   - 新闻问题 → 列出 3-5 条最相关的新闻，每条一行。
   - 比较问题 → 用 Markdown 表格直接对比指标。
5. 用中文回答，专业但易读。"""


FEW_SHOT_USER = """资料：
## 宁德时代（300750.SZ）
- 申万一级：电力设备　二级：电池

### 业绩（最近4季）
| 报告期 | EPS（元） | ROE % | 毛利率 % |
|---|---|---|---|
| 20260331 | 4.58 | 5.97 | 24.82 |
| 20251231 | 16.14 | 24.72 | 26.27 |
| 20250930 | 11.02 | 17.48 | 25.31 |
| 20250630 | 6.92 | 11.25 | 25.02 |

问题：宁德时代最近一季业绩如何？"""

FEW_SHOT_ASSISTANT = """**宁德时代（300750.SZ）2026Q1 业绩摘要**

最新一季报告期 2026-03-31，EPS 4.58 元、ROE 5.97%、毛利率 24.82%。

- ROE 较 2025Q4 单季水平回落（24.72% → 5.97%），主要是单季 vs 累计口径差异，不构成趋势恶化。
- 毛利率 24.82%，相比 2025Q4 的 26.27% 略降，但仍处于近 4 季稳定区间（25%-26%）。
- 单季 EPS 4.58 元，大幅高于 2025 同期，盈利能力保持。

资料未提供 2026Q1 营收 YoY 与归母净利 YoY 的具体数值。"""


class QAEngine:
    def __init__(self,
                 model_id: str = 'Qwen/Qwen2.5-7B-Instruct',
                 device: str = 'cuda',
                 quant: str = '4bit',           # '4bit' for 4070, 'none' for 5090
                 max_new_tokens: int = 600,
                 temperature: float = 0.3):
        import threading
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Serialise GPU access. FastAPI runs sync handlers in a threadpool;
        # without this lock two near-simultaneous /ask calls each spawn a
        # model.generate, doubling the KV-cache and OOM-ing 12 GB cards.
        # The lock is ALSO held across streaming generation so the second
        # request waits for the first to finish before touching the GPU.
        self._gpu_lock = threading.Lock()

        print(f"[qa] loading {model_id} (quant={quant}) ...", flush=True)
        kw = dict(torch_dtype=torch.bfloat16, device_map=device,
                  trust_remote_code=True)
        # NOTE: attn_implementation='sdpa' was tried here for memory
        # efficiency, but it silently hung generate() on the bnb 4-bit
        # Qwen 7B build (model loaded, GPU utilisation stayed at 0 %,
        # /ask never returned). Leaving it as default.
        if quant == '4bit':
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
            )
            kw.pop('torch_dtype')
        elif quant == '8bit':
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            kw.pop('torch_dtype')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model     = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        print(f"[qa] model ready on {self.model.device}", flush=True)

    def _build_messages(self, question: str, context: str) -> list:
        return [
            {'role': 'system',    'content': SYSTEM_PROMPT},
            {'role': 'user',      'content': FEW_SHOT_USER},
            {'role': 'assistant', 'content': FEW_SHOT_ASSISTANT},
            {'role': 'user',      'content': f"资料：\n{context}\n\n问题：{question}"},
        ]

    def _gen_kwargs(self, inputs, max_new_tokens: int) -> dict:
        return dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 0.01),
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, question: str, context: str,
                  max_new_tokens: int | None = None) -> str:
        max_new = max_new_tokens or self.max_new_tokens
        messages = self._build_messages(question, context)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # Cap prompt length aggressively — KV-cache scales linearly with
        # seq_len. 4096 = ~1.5 GB cache headroom on 12 GB during gen.
        # 4 K is the safe cap on a 12 GB card. The peak VRAM during the
        # prompt forward pass is dominated by the attention softmax tensor
        # — shape [heads, seq, seq] @ fp16 = 28 × 6144² × 2 ≈ 2.1 GB at
        # 6 K but only ~0.9 GB at 4 K. SDPA shrinks this further but the
        # cap is the safety net.
        inputs = self.tokenizer(prompt, return_tensors='pt',
                                 truncation=True, max_length=4096).to(self.model.device)
        prompt_len = inputs['input_ids'].shape[1]

        with self._gpu_lock, self.torch.no_grad():
            try:
                out = self.model.generate(**self._gen_kwargs(inputs, max_new))
                new_tokens = out[0, prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            finally:
                # Release transient activation / KV-cache buffers — without
                # this they linger and a series of long-context queries
                # gradually squeezes out room for future generations.
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
        return text

    def generate_stream(self, question: str, context: str,
                         max_new_tokens: int | None = None) -> Iterator[str]:
        """Yield decoded text chunks as they're generated. Drops perceived
        latency from ~15-20 s to <1 s for first tokens.

        The GPU lock is held for the duration of generation — concurrent
        callers wait. This prevents two streams from both being in-flight
        on the GPU at once (which doubles KV-cache and OOMs 12 GB).
        """
        from transformers import TextIteratorStreamer
        max_new = max_new_tokens or self.max_new_tokens
        messages = self._build_messages(question, context)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # 4 K is the safe cap on a 12 GB card. The peak VRAM during the
        # prompt forward pass is dominated by the attention softmax tensor
        # — shape [heads, seq, seq] @ fp16 = 28 × 6144² × 2 ≈ 2.1 GB at
        # 6 K but only ~0.9 GB at 4 K. SDPA shrinks this further but the
        # cap is the safety net.
        inputs = self.tokenizer(prompt, return_tensors='pt',
                                 truncation=True, max_length=4096).to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer,
                                         skip_prompt=True,
                                         skip_special_tokens=True)
        kwargs = self._gen_kwargs(inputs, max_new)
        kwargs['streamer'] = streamer

        with self._gpu_lock:
            thread = Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            try:
                for piece in streamer:
                    if piece:
                        yield piece
            finally:
                # Drain any remaining tokens then join — never leak the
                # background generate thread, otherwise the next request
                # holds the lock forever.
                for _ in streamer:
                    pass
                thread.join()
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quant', choices=['none','8bit','4bit'], default='4bit',
                   help='4bit fits 12GB VRAM; none requires ≥18GB')
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--questions', nargs='+', default=[
        '300750.SZ最近一季度的业绩怎么样？',
        '茅台最近有什么新闻？',
        '比亚迪和宁德时代谁的毛利率更高？',
    ])
    args = p.parse_args()

    r  = Retriever('stock_data/qa/aliases.json',
                    'stock_data/qa/news_linked.parquet')
    cb = ContextBuilder('stock_data/qa/aliases.json')
    qa = QAEngine(model_id=args.model, quant=args.quant)

    for q in args.questions:
        print()
        print('=' * 70)
        print(f'Q: {q}')
        print('=' * 70)
        out = qa.ask(q, r, cb)
        print(f"resolved ts_codes: {out['ts_codes']}  "
              f"({out['n_articles']} articles, {out['context_chars']} ctx chars)")
        print()
        print(out['answer'])


if __name__ == '__main__':
    main()
