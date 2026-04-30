"""
Batch link the deduped news corpus to ts_codes via the Aho-Corasick matcher.

Reads:   stock_data/news_corpus_dedup.parquet  (1.87M rows: source, datetime,
         title, content)
Writes:  stock_data/qa/news_linked.parquet     (same rows + ts_codes_pred:
         list[str]; only rows with ≥1 hit are kept)

Coverage target: lift from ~3.6 % (regex) to ≥ 30 % via aliases. Precision
is high by construction (lexical match against full-name / short-name /
6-digit symbol; no fuzzy match yet).

Usage:
    ./venv/Scripts/python -m qa.linker.predict
    ./venv/Scripts/python -m qa.linker.predict --in_parquet ... --out_parquet ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qa.linker.ahocorasick_matcher import build_matcher, match_article


def _dense_augment(df: pd.DataFrame, lex_matches, args) -> list:
    """Add dense-retrieval candidates for articles where lexical matching
    found 0 codes. Returns the merged ts_codes list per article.

    This is the Phase 2D path — only invoked under --use_dense, and
    only for the subset of articles that need it (so the cost stays
    sub-linear in the corpus size).
    """
    from qa.linker.dense_matcher import DenseEntityMatcher

    # Articles to augment: those with no lexical hits
    miss_idx = [i for i, m in enumerate(lex_matches) if not m]
    print(f"[link] dense augment: {len(miss_idx):,} articles "
          f"({100*len(miss_idx)/max(len(df),1):.1f} %) need help")
    if not miss_idx:
        return list(lex_matches)

    # Build texts
    title_arr   = df['title'].fillna('').astype(str).values
    content_arr = df['content'].fillna('').astype(str).values
    texts = [
        f"{title_arr[i]}。 {content_arr[i][:args.dense_max_content]}"
        for i in miss_idx
    ]

    dm = DenseEntityMatcher(args.entity_faiss, args.entity_meta)

    # Optional reranker bundle
    reranker_bundle = None
    if Path(args.reranker).exists():
        with open(args.reranker, 'rb') as f:
            reranker_bundle = pickle.load(f)
        print(f"[link] loaded reranker: {args.reranker}")
    else:
        print(f"[link] no reranker found at {args.reranker} — "
              f"falling back to dense-cos threshold {args.dense_cos_thresh}")

    with open(args.aliases, 'r', encoding='utf-8') as f:
        alias_dict = json.load(f)
    ts_aliases = {ts: v.get('aliases', []) for ts, v in alias_dict.items()}

    out = list(lex_matches)
    t0 = time.time()
    n_added = 0

    for chunk_start in range(0, len(miss_idx), args.dense_batch * 8):
        chunk_idxs = miss_idx[chunk_start: chunk_start + args.dense_batch * 8]
        chunk_texts = [texts[chunk_start + j] for j in range(len(chunk_idxs))]
        cands_batch = dm.search(chunk_texts, top_k=args.dense_top_k,
                                 batch_size=args.dense_batch)
        for i_local, i_global in enumerate(chunk_idxs):
            cands = cands_batch[i_local]
            ts_picked = []
            if reranker_bundle is not None:
                # Score each candidate with the LR
                from qa.linker.train_reranker import _features, FEATURE_NAMES
                X_rows = []
                for ts, cos in cands:
                    X_rows.append(_features(
                        content_arr[i_global][:args.dense_max_content],
                        title_arr[i_global], ts, cos, ts_aliases,
                    ))
                if X_rows:
                    proba = reranker_bundle['model'].predict_proba(np.array(X_rows))[:, 1]
                    for (ts, _cos), p in zip(cands, proba):
                        if p >= args.rerank_thresh:
                            ts_picked.append(ts)
            else:
                # Threshold on cosine alone
                for ts, cos in cands:
                    if cos >= args.dense_cos_thresh:
                        ts_picked.append(ts)
            # Cap at 3 to limit noise
            ts_picked = ts_picked[:3]
            if ts_picked:
                out[i_global] = ts_picked
                n_added += 1
        if chunk_start and (chunk_start // args.dense_batch) % 50 == 0:
            done = chunk_start + len(chunk_idxs)
            rate = done / max(time.time() - t0, 1e-9)
            eta_min = (len(miss_idx) - done) / max(rate, 1e-9) / 60
            print(f"  dense  [{done:,}/{len(miss_idx):,}]  "
                  f"{rate:.0f} art/s  ETA {eta_min:.1f} min", flush=True)
    print(f"[link] dense added matches to {n_added:,} previously-missed articles")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_parquet',  default='stock_data/news_corpus_dedup.parquet')
    p.add_argument('--out_parquet', default='stock_data/qa/news_linked.parquet')
    p.add_argument('--aliases',     default='stock_data/qa/aliases.json')
    p.add_argument('--keep_unlinked', action='store_true',
                   help='if set, also keep rows with 0 stock matches '
                        '(default: drop to slim the output)')
    p.add_argument('--use_dense', action='store_true',
                   help='Augment lexical matches with bge-m3 dense retrieval '
                        '+ optional LR reranker for articles missed by AC.')
    p.add_argument('--entity_faiss', default='stock_data/qa/entities.faiss')
    p.add_argument('--entity_meta',  default='stock_data/qa/entities.parquet')
    p.add_argument('--reranker',     default='stock_data/qa/linker_reranker.pkl')
    p.add_argument('--dense_top_k',  type=int, default=10)
    p.add_argument('--dense_batch',  type=int, default=64)
    p.add_argument('--dense_max_content', type=int, default=512)
    p.add_argument('--dense_cos_thresh', type=float, default=0.65,
                   help='Cosine threshold when no reranker is available.')
    p.add_argument('--rerank_thresh', type=float, default=0.5,
                   help='LR predict_proba threshold for accepting a candidate.')
    args = p.parse_args()

    in_p  = Path(args.in_parquet)
    out_p = Path(args.out_parquet)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"[link] loading {in_p} ...")
    df = pd.read_parquet(in_p)
    print(f"[link] {len(df):,} articles loaded")

    matcher = build_matcher(args.aliases)

    print(f"[link] matching ...")
    t0 = time.time()
    title_arr   = df['title'].fillna('').astype(str).values
    content_arr = df['content'].fillna('').astype(str).values
    matches = []
    for i in range(len(df)):
        if i and i % 50000 == 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            print(f"  [{i:>9,}/{len(df):,}]  {rate:.0f} art/s  "
                  f"ETA {(len(df) - i) / max(rate, 1e-6) / 60:.1f} min",
                  flush=True)
        matches.append(match_article(matcher, title_arr[i], content_arr[i]))

    if args.use_dense:
        matches = _dense_augment(df, matches, args)

    df['ts_codes_pred'] = matches
    df['n_codes_pred']  = df['ts_codes_pred'].str.len()

    n_with_match = (df['n_codes_pred'] > 0).sum()
    out = df if args.keep_unlinked else df[df['n_codes_pred'] > 0]

    out.to_parquet(out_p, index=False)
    print()
    print("=" * 60)
    print(f"LINK SUMMARY")
    print("=" * 60)
    print(f"  total articles      : {len(df):,}")
    print(f"  with ≥1 match       : {n_with_match:,}  "
          f"({100*n_with_match/max(len(df),1):.1f} %)")
    print(f"  output              : {out_p}  ({out_p.stat().st_size / 1e6:.1f} MB)")
    print(f"  total time          : {(time.time() - t0) / 60:.1f} min")
    if n_with_match:
        print(f"  avg codes/article   : "
              f"{df.loc[df['n_codes_pred']>0, 'n_codes_pred'].mean():.2f}")
        print(f"  median codes/article: "
              f"{df.loc[df['n_codes_pred']>0, 'n_codes_pred'].median():.0f}")


if __name__ == '__main__':
    main()
