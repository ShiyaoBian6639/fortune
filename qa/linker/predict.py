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
import time
from pathlib import Path

import pandas as pd

from qa.linker.ahocorasick_matcher import build_matcher, match_article


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_parquet',  default='stock_data/news_corpus_dedup.parquet')
    p.add_argument('--out_parquet', default='stock_data/qa/news_linked.parquet')
    p.add_argument('--aliases',     default='stock_data/qa/aliases.json')
    p.add_argument('--keep_unlinked', action='store_true',
                   help='if set, also keep rows with 0 stock matches '
                        '(default: drop to slim the output)')
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
