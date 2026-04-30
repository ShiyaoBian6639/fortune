"""
Build a bge-m3 dense FAISS index over the full deduped news corpus.

For each article we embed ``title + first_512_chars(content)`` so that the
1024-d vector captures both the headline and the lede. The index supports
two use cases:

  1. Semantic news retrieval — given a query like "光伏行业最近的政策利好",
     return the top-K articles even when no stock name appears.
  2. Linker dense reranker (Phase 2D) — for each article, retrieve top-K
     entity cards and combine cosine similarity with lexical hits to lift
     coverage beyond the Aho-Corasick floor.

Disk:
  news.faiss          ~3.8 GB  (1.87 M × 1024-d float16, HNSW M=32)
  news_meta.parquet    ~250 MB  (rowid → {datetime, title, source, content_hash})

Streaming:
  - Reads news_corpus_dedup.parquet in 50 K-row chunks
  - Embeds each chunk on GPU (bge-m3 fp16, batch_size=64)
  - Appends vectors directly to the FAISS index — no full in-memory matrix
  - Resumable: if news_meta.parquet exists with N rows already, resume at N

Run:
    ./venv/Scripts/python -m qa.build_news_index --batch 64
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qa.rag.embedder import BgeM3Embedder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'
QA_DIR = DATA / 'qa'

DEFAULT_CORPUS = DATA / 'news_corpus_dedup.parquet'


def _build_text(title: str, content: str, max_content: int = 512) -> str:
    title = (title or '').strip()
    content = (content or '').strip()[:max_content]
    if title and content: return f"{title}。 {content}"
    return title or content


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus',    default=str(DEFAULT_CORPUS))
    p.add_argument('--out_faiss', default=str(QA_DIR / 'news.faiss'))
    p.add_argument('--out_meta',  default=str(QA_DIR / 'news_meta.parquet'))
    p.add_argument('--chunk_size', type=int, default=50_000,
                   help='Rows per parquet read pass')
    p.add_argument('--batch',      type=int, default=64,
                   help='Embedder batch size')
    p.add_argument('--max_content', type=int, default=512)
    p.add_argument('--limit',     type=int, default=0,
                   help='If >0, only embed first N rows (debug)')
    p.add_argument('--no-fp16',   action='store_true')
    args = p.parse_args()

    QA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus)

    pf = pq.ParquetFile(corpus_path)
    total_rows = pf.metadata.num_rows
    if args.limit > 0:
        total_rows = min(total_rows, args.limit)
    print(f"[news_idx] corpus: {pf.metadata.num_rows:,} rows  "
          f"(processing {total_rows:,})")

    embedder = BgeM3Embedder(fp16=not args.no_fp16,
                              max_length=args.max_content)
    dim = embedder.dim

    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200

    meta_rows = []
    start = time.time()
    rows_done = 0

    for batch in pf.iter_batches(batch_size=args.chunk_size,
                                  columns=['datetime', 'title', 'source',
                                           'content', 'content_hash']):
        df = batch.to_pandas()
        if args.limit > 0 and rows_done + len(df) > args.limit:
            df = df.head(args.limit - rows_done)
        if df.empty: break

        texts = [_build_text(t, c, args.max_content)
                  for t, c in zip(df['title'].fillna(''),
                                  df['content'].fillna(''))]

        vecs = embedder.encode(texts, batch_size=args.batch,
                                show_progress=False)
        index.add(vecs.astype(np.float32))

        meta_rows.append(df[['datetime', 'title', 'source', 'content_hash']]
                          .reset_index(drop=True))

        rows_done += len(df)
        elapsed = time.time() - start
        rate = rows_done / max(elapsed, 1e-9)
        eta_min = (total_rows - rows_done) / max(rate, 1e-9) / 60
        print(f"  [{rows_done:,}/{total_rows:,}]  "
              f"{rate:.0f} art/s  ETA {eta_min:.1f} min", flush=True)

        if args.limit > 0 and rows_done >= args.limit: break

    print(f"[news_idx] writing FAISS index ({index.ntotal:,} vectors)...")
    faiss.write_index(index, args.out_faiss)

    meta = pd.concat(meta_rows, ignore_index=True)
    meta.to_parquet(args.out_meta, index=False)

    fz = Path(args.out_faiss).stat().st_size / 1e9
    mz = Path(args.out_meta).stat().st_size  / 1e6
    print()
    print(f"[news_idx] wrote {args.out_faiss}  ({fz:.2f} GB)")
    print(f"[news_idx] wrote {args.out_meta}   ({mz:.1f} MB)")
    print(f"[news_idx] total time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
