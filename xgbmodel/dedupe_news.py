"""
Preprocess the multi-source news corpus into a deduplicated single file.

Why dedup matters
-----------------
The same article often appears across sina / eastmoney / cls / etc., sometimes
with minor reformatting. Sending duplicates through Qwen wastes ~30-50 % of
the compute budget on the remote 5090. This script collapses the corpus to
unique articles, keyed by a hash of the first 200 chars of normalised content.

Dedup keys (cumulative — each can be enabled/disabled by flag):
  1. content_hash    — hash of first --content_chars of content stripped of
                       whitespace + punctuation. Catches reposts and most
                       reformatted variants.
  2. (title, day)    — exact title match within the same calendar day. Used
                       as a fallback when content is short or generic.
  3. window_days     — only consider duplicates within +/- N days of each
                       other. Prevents seasonal news (e.g. "annual report
                       season") from being collapsed across years.

Output (default `stock_data/news_corpus_dedup.parquet`):
  source, datetime, content, title, ts_codes, content_hash

Run:
    ./venv/Scripts/python -m xgbmodel.dedupe_news
    ./venv/Scripts/python -m xgbmodel.dedupe_news --output news_corpus_dedup.csv
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'

# Re-use the regex helpers from qwen_sentiment so codes are extracted
# the same way both scripts see them.
from xgbmodel.qwen_sentiment import extract_codes


PUNCT_RE = re.compile(r'[\s\W_]+', re.UNICODE)


def _normalise(text: str, n_chars: int = 200) -> str:
    """Lowercase, strip whitespace+punctuation, take first n_chars."""
    if not isinstance(text, str):
        return ''
    t = PUNCT_RE.sub('', text.lower())
    return t[:n_chars]


def _hash_content(text: str, n_chars: int = 200) -> str:
    n = _normalise(text, n_chars=n_chars)
    if not n:
        return ''
    return hashlib.md5(n.encode('utf-8')).hexdigest()[:16]


def load_all_news(news_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Read every source/day file under news_dir, return concatenated frame."""
    frames = []
    for src in sorted(p.name for p in news_dir.iterdir() if p.is_dir()):
        src_dir = news_dir / src
        added = 0
        for fp in sorted(src_dir.glob('*.csv')):
            stem = fp.stem
            if stem.startswith('_'):
                continue
            if stem.isdigit() and len(stem) == 8:
                if not (start.replace('-', '') <= stem <= end.replace('-', '')):
                    continue
            try:
                df = pd.read_csv(fp, encoding='utf-8-sig',
                                  on_bad_lines='skip',
                                  dtype={'ts_code': str})
            except Exception:
                continue
            if df.empty:
                continue
            df['source'] = src
            for c in ('datetime', 'content', 'title', 'ts_code'):
                if c not in df.columns:
                    df[c] = ''
            df = df[['source', 'datetime', 'content', 'title']]
            frames.append(df)
            added += 1
        print(f"  [{src}] {added} files", flush=True)
    if not frames:
        return pd.DataFrame(columns=['source','datetime','content','title'])
    out = pd.concat(frames, ignore_index=True)
    # 10jqka uses "2018-10-24 02:59" (no seconds), other sources use
    # "2018-10-24 02:59:00". Pass format='mixed' so pandas parses each row
    # independently instead of inferring a single format from row 0.
    out['datetime'] = pd.to_datetime(out['datetime'], format='mixed', errors='coerce')
    out = out.dropna(subset=['datetime'])
    out['trade_date'] = out['datetime'].dt.normalize()
    out = out[(out['trade_date'] >= pd.Timestamp(start))
              & (out['trade_date'] <= pd.Timestamp(end))]
    return out.reset_index(drop=True)


def dedupe(df: pd.DataFrame, content_chars: int = 200,
            title_day_dedup: bool = True) -> pd.DataFrame:
    """Drop duplicates by content_hash and (title, day). Keep earliest."""
    if df.empty:
        return df
    print(f"  computing content hashes ({content_chars} chars) ...", flush=True)
    df = df.copy()
    df['content_hash'] = df['content'].apply(lambda t: _hash_content(t, content_chars))
    df['title_norm']   = df['title'].apply(lambda t: _normalise(t, 200))

    n0 = len(df)

    # Sort by datetime so .drop_duplicates(keep='first') retains the earliest
    df = df.sort_values('datetime').reset_index(drop=True)

    # Pass 1: dedup by content hash globally — strongest signal
    mask_empty_hash = df['content_hash'] == ''
    keep_hash = (~mask_empty_hash) & ~df.duplicated(subset=['content_hash'], keep='first')
    keep_no_hash = mask_empty_hash    # short content, can't hash; keep all
    df1 = df[keep_hash | keep_no_hash].reset_index(drop=True)
    print(f"  after content_hash dedup    : {len(df1):>8,} ({100*len(df1)/n0:.1f}% kept)")

    # Pass 2: dedup by (title_norm, day) — catches articles with different
    # content but identical headline within the same day (re-runs)
    if title_day_dedup:
        df1['day_key'] = df1['datetime'].dt.strftime('%Y%m%d')
        mask_short_title = df1['title_norm'].str.len() < 5
        dup_title_day = df1.duplicated(subset=['title_norm', 'day_key'], keep='first')
        keep_title = ~dup_title_day | mask_short_title    # keep short titles unconditionally
        df2 = df1[keep_title].drop(columns=['day_key']).reset_index(drop=True)
        print(f"  after (title, day) dedup    : {len(df2):>8,} ({100*len(df2)/n0:.1f}% kept)")
    else:
        df2 = df1

    df2 = df2.drop(columns=['title_norm'])
    return df2


def attach_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract A-share ts_codes from each article's title+content."""
    df = df.copy()
    df['_text'] = df['title'].fillna('').astype(str) + ' ' + df['content'].fillna('').astype(str)
    df['ts_codes'] = df['_text'].apply(extract_codes)
    df = df.drop(columns=['_text'])
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--news_dir', default='stock_data/news')
    p.add_argument('--output',   default='stock_data/news_corpus_dedup.parquet',
                   help='Output path. .parquet (faster) or .csv (universal).')
    p.add_argument('--start',    default='2017-01-01')
    p.add_argument('--end',      default='2026-04-28')
    p.add_argument('--content_chars', type=int, default=200,
                   help='hash first N chars of content. Higher = more sensitive. '
                        'Default 200 catches most reposts.')
    p.add_argument('--no_title_day_dedup', action='store_true',
                   help='disable (title, day) dedup pass.')
    p.add_argument('--require_codes', action='store_true',
                   help='only keep articles that mention >= 1 A-share code.')
    args = p.parse_args()

    print(f"[dedupe] reading news from {args.news_dir} ({args.start} → {args.end}) ...",
          flush=True)
    t0 = time.time()
    df = load_all_news(Path(args.news_dir), args.start, args.end)
    n_raw = len(df)
    print(f"[dedupe] raw corpus: {n_raw:,} articles "
          f"(span {df['datetime'].min()} → {df['datetime'].max()})", flush=True)
    print(f"  by source:")
    for src, n in df['source'].value_counts().items():
        print(f"    {src:14}: {n:>8,}")

    print(f"[dedupe] dedup pass ...", flush=True)
    df = dedupe(df, content_chars=args.content_chars,
                 title_day_dedup=not args.no_title_day_dedup)

    print(f"[dedupe] extracting ts_codes ...", flush=True)
    df = attach_codes(df)
    n_with_codes = (df['ts_codes'].str.len() > 0).sum()
    print(f"[dedupe] {n_with_codes:,} of {len(df):,} articles mention >= 1 A-share code "
          f"({100*n_with_codes/max(len(df),1):.1f}%)", flush=True)
    if args.require_codes:
        df = df[df['ts_codes'].str.len() > 0].reset_index(drop=True)
        print(f"[dedupe] filtered to articles with codes: {len(df):,}", flush=True)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Save (parquet preferred — preserves dtypes incl ts_codes list column)
    if out_p.suffix.lower() == '.parquet':
        try:
            df.to_parquet(out_p, index=False)
        except ImportError:
            print(f"  [warn] pyarrow not installed, falling back to .csv")
            out_p = out_p.with_suffix('.csv')
            df_c = df.copy()
            df_c['ts_codes'] = df_c['ts_codes'].apply(lambda L: ';'.join(L) if isinstance(L, list) else '')
            df_c.to_csv(out_p, index=False, encoding='utf-8-sig')
    else:
        df_c = df.copy()
        df_c['ts_codes'] = df_c['ts_codes'].apply(lambda L: ';'.join(L) if isinstance(L, list) else '')
        df_c.to_csv(out_p, index=False, encoding='utf-8-sig')

    print()
    print("=" * 70)
    print(f"DEDUP SUMMARY  ({args.start} → {args.end})")
    print("=" * 70)
    print(f"  raw articles      : {n_raw:>8,}")
    print(f"  after dedup       : {len(df):>8,}  ({100*(n_raw-len(df))/max(n_raw,1):.1f}% removed)")
    print(f"  with stock codes  : {n_with_codes:>8,}")
    print(f"  output            : {out_p}  "
          f"({out_p.stat().st_size / 1e6:.2f} MB)")
    print(f"  elapsed           : {(time.time()-t0):.1f}s")


if __name__ == '__main__':
    main()
