"""
Standalone news-sentiment scorer using Qwen2.5-7B-Instruct.

Designed to run on a remote workstation (e.g. RTX 5090 with ≥24 GB VRAM) —
this script has no dependencies on the rest of the xgbmodel pipeline. It reads
news CSVs, regex-extracts mentioned A-share ts_codes, batches articles
through Qwen, and writes a sentiment table that the local pipeline picks up.

Usage on remote:
    git pull
    pip install -r requirements_qwen.txt          # see bottom of file
    ./venv/Scripts/python -m xgbmodel.qwen_sentiment \\
        --news_dir stock_data/news \\
        --out_csv  stock_data/news_sentiment_qwen.csv \\
        --start    2017-01-01 --end 2026-04-27 \\
        --device   cuda --batch_size 16

Then push the resulting CSV back:
    git add stock_data/news_sentiment_qwen.csv
    git commit -m "Qwen sentiment: 2017-2026"
    git push

The local machine then pulls + uses news_sentiment_qwen.csv as a feature.

Output schema:
    ts_code, trade_date, n_articles, sentiment_mean, sentiment_pos_share,
    sentiment_neg_share, sentiment_neu_share, top_keywords

Hardware notes
--------------
  • Qwen2.5-7B-Instruct in bfloat16 = ~14 GB VRAM (fits 5090 32 GB easily,
    leaves headroom for KV cache).
  • For RTX 4070 Super (12 GB), use --quant 4bit (bitsandbytes 4-bit). Slower
    but functional. The default `--quant none` requires ≥18 GB.
  • Throughput: ~30 tok/s/article on 5090 in bf16. With 5-token prompts +
    50-token responses, ~30 articles/sec. A 5-year news corpus (~500K
    articles) takes ~4 hours.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ─── Stock-code regex ───────────────────────────────────────────────────────
# Matches "600000.SH", "000001.SZ", "300750", or just a 6-digit code in
# context. We require the leading digit to match A-share conventions:
#   6 → SSE main board    (.SH)
#   0 → SZSE main board   (.SZ)
#   3 → ChiNext / SZSE    (.SZ)
#   8 → BJSE              (.BJ)  — excluded per project convention
RE_TS_FULL = re.compile(r'\b([036]\d{5})\.(?:SH|SZ)\b')
RE_TS_BARE = re.compile(r'(?<![0-9])([036]\d{5})(?![0-9])')


def code_to_ts(code: str) -> str:
    """Append .SH / .SZ to a 6-digit code based on its leading digit."""
    if code.startswith('6'):     return f'{code}.SH'
    if code.startswith(('0','3')): return f'{code}.SZ'
    return ''


def extract_codes(text: str) -> List[str]:
    """Return the list of ts_codes mentioned in text (deduped, ordered)."""
    seen, out = set(), []
    if not isinstance(text, str): return out
    for m in RE_TS_FULL.finditer(text):
        ts = f'{m.group(1)}.{text[m.end()-2:m.end()]}' if False else \
             f'{m.group(1)}.{m.group(0)[-2:]}'
        if ts not in seen:
            seen.add(ts); out.append(ts)
    for m in RE_TS_BARE.finditer(text):
        ts = code_to_ts(m.group(1))
        if ts and ts not in seen:
            seen.add(ts); out.append(ts)
    return out


# ─── News loading ────────────────────────────────────────────────────────────
def load_deduped_corpus(path: Path, start: str, end: str) -> pd.DataFrame:
    """Read the pre-deduped corpus (built by xgbmodel.dedupe_news).
    Skips the per-source rescan + dedup that load_news_corpus does, which on a
    full 2017-now corpus saves ~5 min and ~30-50% of articles."""
    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding='utf-8-sig')
        if 'ts_codes' in df.columns and df['ts_codes'].dtype == object:
            df['ts_codes'] = df['ts_codes'].fillna('').apply(
                lambda s: s.split(';') if isinstance(s, str) and s else [])
    df['datetime']   = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['trade_date'] = df['datetime'].dt.normalize()
    df = df[(df['trade_date'] >= pd.Timestamp(start))
            & (df['trade_date'] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


def load_news_corpus(news_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Read every per-source-per-day CSV in news_dir and return a unified frame.

    Per-stock news files (e.g. sh/600000_news.csv) are also loaded if present.
    Output columns: source, datetime, content, title, ts_codes (list).
    """
    frames = []
    for src in sorted(p.name for p in news_dir.iterdir() if p.is_dir()):
        src_dir = news_dir / src
        for fp in sorted(src_dir.glob('*.csv')):
            stem = fp.stem
            # Daily files like 20180101.csv
            if stem.isdigit() and len(stem) == 8:
                if not (start.replace('-','') <= stem <= end.replace('-','')):
                    continue
            try:
                df = pd.read_csv(fp, encoding='utf-8-sig',
                                  on_bad_lines='skip',
                                  dtype={'ts_code': str})
            except Exception:
                continue
            if df.empty: continue
            df['source'] = src
            for c in ('datetime','content','title','ts_code','stock_name'):
                if c not in df.columns: df[c] = ''
            df = df[['source','datetime','content','title','ts_code']]
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['source','datetime','content','title','ts_code'])
    out = pd.concat(frames, ignore_index=True)
    out['datetime'] = pd.to_datetime(out['datetime'], format='mixed', errors='coerce')
    out = out.dropna(subset=['datetime'])
    out['trade_date'] = out['datetime'].dt.normalize()
    out = out[(out['trade_date'] >= pd.Timestamp(start))
              & (out['trade_date'] <= pd.Timestamp(end))]
    return out.reset_index(drop=True)


# ─── Qwen model wrapper ─────────────────────────────────────────────────────
class QwenSentiment:
    """Wraps Qwen2.5-7B-Instruct for batched 3-class sentiment classification."""

    SYSTEM_PROMPT = (
        "你是一名专业的中国A股市场情绪分析师。"
        "对每篇新闻给出三选一的情绪标签：positive / negative / neutral。"
        "positive = 该新闻对所提及个股或市场短期走势利好；"
        "negative = 利空；neutral = 信息中立或与股价短期走势无明显关联。"
        "只输出一个小写英文单词标签，不输出解释。"
    )

    def __init__(self, model_id: str = 'Qwen/Qwen2.5-7B-Instruct',
                 device: str = 'cuda', quant: str = 'none',
                 max_input_chars: int = 600):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.torch = torch
        self.device = device
        self.max_input_chars = max_input_chars

        print(f"[qwen] loading {model_id} (quant={quant}) ...", flush=True)
        kw = dict(torch_dtype=torch.bfloat16, device_map=device,
                  trust_remote_code=True)
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
        print(f"[qwen] model ready on {self.model.device}", flush=True)

        # Pre-tokenise the system prompt once
        self._sys_msg = {'role': 'system', 'content': self.SYSTEM_PROMPT}

        # Pre-compute token ids for the 3 expected labels (single-token greedy
        # decoding is only valid if each label is one token; "positive" /
        # "negative" / "neutral" tokenise to 1 token each in Qwen's vocab).
        self.label_ids = {}
        for lbl in ('positive', 'negative', 'neutral'):
            ids = self.tokenizer.encode(lbl, add_special_tokens=False)
            self.label_ids[lbl] = ids[0]      # use first token
        print(f"[qwen] label token ids: {self.label_ids}", flush=True)

    @torch.no_grad() if False else (lambda f: f)   # placeholder; real decorator below
    def score_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Return list of (label, prob) tuples. Use first-token logits for speed."""
        import torch
        # Build chat-format prompts
        prompts = []
        for t in texts:
            t = (t or '')[:self.max_input_chars]
            messages = [self._sys_msg,
                        {'role': 'user', 'content': f'新闻：{t}\n标签：'}]
            p = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompts.append(p)

        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True,
                                 truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outs = self.model(**inputs, return_dict=True)
            logits = outs.logits

        # For each row, take the logits at the LAST non-pad position (where the
        # next token would be generated). Then look up the 3 label tokens.
        attn = inputs['attention_mask']
        last_idx = attn.sum(dim=1) - 1                         # (B,)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_idx, last_idx]              # (B, V)
        results = []
        label_ids = list(self.label_ids.values())
        label_names = list(self.label_ids.keys())
        sub_logits = last_logits[:, label_ids]                 # (B, 3)
        probs = torch.softmax(sub_logits.float(), dim=-1).cpu().numpy()
        for i in range(len(texts)):
            j = int(np.argmax(probs[i]))
            results.append((label_names[j], float(probs[i, j])))
        return results


# ─── Main aggregation pipeline ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--news_dir', default='stock_data/news')
    p.add_argument('--dedup_corpus', default='stock_data/news_corpus_dedup.parquet',
                   help='preferred input — built by xgbmodel.dedupe_news. If '
                        'absent, falls back to scanning per-source files.')
    p.add_argument('--out_csv',  default='stock_data/news_sentiment_qwen.csv')
    p.add_argument('--start',    default='2017-01-01')
    p.add_argument('--end',      default='2026-04-27')
    p.add_argument('--model_id', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--device',   default='cuda')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--quant',    choices=['none','8bit','4bit'], default='none',
                   help='quantization. 4bit fits 12GB VRAM; none recommended for ≥18GB.')
    p.add_argument('--max_articles', type=int, default=0,
                   help='cap total articles for testing; 0 = no cap.')
    p.add_argument('--max_input_chars', type=int, default=600,
                   help='truncate article body before tokenization. 600 ≈ 2-3 sentences.')
    p.add_argument('--checkpoint_every', type=int, default=5000,
                   help='save partial output every N articles.')
    args = p.parse_args()

    # Prefer the deduped corpus if present — saves rescanning + 30-50% compute.
    dedup_p = Path(args.dedup_corpus) if args.dedup_corpus else None
    if dedup_p and dedup_p.exists():
        print(f"[qwen-sent] loading DEDUPED corpus from {dedup_p} ...", flush=True)
        corpus = load_deduped_corpus(dedup_p, args.start, args.end)
    else:
        print(f"[qwen-sent] loading news from {args.news_dir} (no dedup file) ...", flush=True)
        corpus = load_news_corpus(Path(args.news_dir), args.start, args.end)
    print(f"[qwen-sent] {len(corpus):,} articles loaded "
          f"({corpus['datetime'].min()} → {corpus['datetime'].max()})", flush=True)
    if args.max_articles > 0:
        corpus = corpus.sample(min(args.max_articles, len(corpus)),
                                random_state=42).reset_index(drop=True)
        print(f"[qwen-sent] downsampled to {len(corpus):,} articles for testing")

    # Combine title + content for richer context
    corpus['text'] = (corpus['title'].fillna('').astype(str) + ' ' +
                      corpus['content'].fillna('').astype(str))
    corpus['text'] = corpus['text'].str.strip()

    # Extract ts_codes from each article — skip if already present (deduped corpus).
    if 'ts_codes' not in corpus.columns or corpus['ts_codes'].apply(
            lambda x: not isinstance(x, list)).any():
        print(f"[qwen-sent] extracting stock codes from articles ...", flush=True)
        corpus['ts_codes'] = corpus['text'].apply(extract_codes)
    else:
        print(f"[qwen-sent] ts_codes already extracted (from deduped corpus)", flush=True)
    n_with_code = (corpus['ts_codes'].str.len() > 0).sum()
    print(f"[qwen-sent] {n_with_code:,} of {len(corpus):,} articles mention "
          f"≥1 A-share code ({100*n_with_code/max(len(corpus),1):.1f}%)", flush=True)

    # Use only articles with at least 1 code (others have no place in per-stock output)
    work = corpus[corpus['ts_codes'].str.len() > 0].reset_index(drop=True)
    if work.empty:
        print("[qwen-sent] no articles with stock codes — exit"); return

    # Initialise model
    qw = QwenSentiment(model_id=args.model_id, device=args.device,
                        quant=args.quant, max_input_chars=args.max_input_chars)

    # Score in batches
    print(f"[qwen-sent] scoring {len(work):,} articles in batches of {args.batch_size} ...", flush=True)
    t0 = time.time()
    labels: List[str] = [None] * len(work)
    confs:  List[float] = [0.0] * len(work)
    for s in range(0, len(work), args.batch_size):
        batch = work['text'].iloc[s:s + args.batch_size].tolist()
        results = qw.score_batch(batch)
        for k, (lbl, conf) in enumerate(results):
            labels[s + k] = lbl
            confs [s + k] = conf
        # Progress + checkpoint
        if (s // args.batch_size) % 50 == 0:
            done = s + len(batch)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta_min = (len(work) - done) / max(rate, 1e-6) / 60
            print(f"  [{done:>7}/{len(work):>7}]  "
                  f"{rate:.1f} art/s  ETA {eta_min:.1f} min", flush=True)
        if args.checkpoint_every > 0 and s > 0 and s % args.checkpoint_every == 0:
            _checkpoint(work[:s + len(batch)], labels[:s + len(batch)],
                         confs[:s + len(batch)], args.out_csv)

    work['sentiment'] = labels
    work['confidence'] = confs

    # ── Per-(ts_code, trade_date) aggregation ──
    print(f"[qwen-sent] aggregating per (ts_code, trade_date) ...", flush=True)
    rows = []
    for _, r in work.iterrows():
        for ts in r['ts_codes']:
            rows.append({'ts_code': ts, 'trade_date': r['trade_date'],
                          'sentiment': r['sentiment'], 'confidence': r['confidence']})
    flat = pd.DataFrame(rows)
    if flat.empty:
        print("[qwen-sent] no per-stock rows — exit"); return

    flat['s_pos'] = (flat['sentiment'] == 'positive').astype(int)
    flat['s_neg'] = (flat['sentiment'] == 'negative').astype(int)
    flat['s_neu'] = (flat['sentiment'] == 'neutral').astype(int)
    flat['s_signed'] = flat['s_pos'] - flat['s_neg']

    g = flat.groupby(['ts_code', 'trade_date'], as_index=False).agg(
        n_articles=('sentiment', 'size'),
        sentiment_mean=('s_signed', 'mean'),
        sentiment_pos_share=('s_pos', 'mean'),
        sentiment_neg_share=('s_neg', 'mean'),
        sentiment_neu_share=('s_neu', 'mean'),
        confidence_mean=('confidence', 'mean'),
    )
    g = g.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    g['trade_date'] = g['trade_date'].dt.strftime('%Y-%m-%d')

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_p, index=False, encoding='utf-8-sig')

    print()
    print("=" * 70)
    print(f"QWEN SENTIMENT  ({args.start} → {args.end})")
    print("=" * 70)
    print(f"  articles processed   : {len(work):,}")
    print(f"  per-stock-day rows   : {len(g):,}")
    print(f"  unique stocks        : {g['ts_code'].nunique():,}")
    print(f"  date span            : {g['trade_date'].min()} → {g['trade_date'].max()}")
    print(f"  positive share       : {flat['s_pos'].mean()*100:.1f}%")
    print(f"  negative share       : {flat['s_neg'].mean()*100:.1f}%")
    print(f"  neutral  share       : {flat['s_neu'].mean()*100:.1f}%")
    print(f"  output               : {out_p}")
    print(f"  total wall time      : {(time.time() - t0) / 60:.1f} min")


def _checkpoint(work_df, labels, confs, out_csv: str):
    """Write a partial output snapshot so a long run is resumable."""
    snap = work_df.copy()
    snap['sentiment'] = labels
    snap['confidence'] = confs
    rows = []
    for _, r in snap.iterrows():
        for ts in r['ts_codes']:
            rows.append({'ts_code': ts, 'trade_date': r['trade_date'],
                          'sentiment': r['sentiment'],
                          'confidence': r['confidence']})
    if not rows: return
    flat = pd.DataFrame(rows)
    out_p = Path(out_csv).with_suffix('.checkpoint.csv')
    flat.to_csv(out_p, index=False, encoding='utf-8-sig')
    print(f"  [checkpoint] {len(flat):,} rows → {out_p}", flush=True)


# ─── End of file. requirements_qwen.txt is shipped alongside this script. ───
if __name__ == '__main__':
    main()
