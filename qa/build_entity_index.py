"""
Build a bge-m3 dense index over per-stock "entity cards".

For each ts_code we render a ~150-token text snapshot containing:
  - market name + ts_code + symbol
  - full company name (com_name)
  - all aliases joined
  - SW L1 / L2 / industry / area
  - chairman + manager
  - index membership tags (CSI300, SSE50, ...)
  - latest fundamentals one-liner (EPS / ROE / 毛利率 / 营收YoY)

These cards capture the semantic surface of each stock so that a query like
"新能源车板块龙头" can semantically match BYD/CATL even when no alias is
present in the question.

Outputs (under ``stock_data/qa/``):
  entities.faiss          FAISS HNSW index (cosine via inner product on
                           L2-normalised vectors), 1024-d
  entities.parquet        Sidecar: row → ts_code + name + card (for debug)

Run:
    ./venv/Scripts/python -m qa.build_entity_index
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from qa.rag.embedder import BgeM3Embedder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'
QA_DIR = DATA / 'qa'


def _latest_fina_snippet(ts_code: str) -> str:
    """One-line fundamentals snapshot from the most recent fina_indicator row."""
    code, suf = ts_code.split('.')
    fp = DATA / 'fina_indicator' / f'{code}_{suf.upper()}.csv'
    if not fp.exists(): return ''
    try:
        df = pd.read_csv(fp, encoding='utf-8-sig', dtype={'end_date': str})
    except Exception:
        return ''
    if df.empty or 'end_date' not in df.columns:
        return ''
    df = df.sort_values('end_date').tail(1).iloc[0]
    parts = []
    if pd.notna(df.get('end_date')):     parts.append(f"截至{df['end_date']}")
    if pd.notna(df.get('eps')):          parts.append(f"EPS {df['eps']:.2f}")
    if pd.notna(df.get('roe')):          parts.append(f"ROE {df['roe']:.2f}%")
    if pd.notna(df.get('grossprofit_margin')):
        parts.append(f"毛利率 {df['grossprofit_margin']:.2f}%")
    if pd.notna(df.get('netprofit_margin')):
        parts.append(f"净利率 {df['netprofit_margin']:.2f}%")
    if pd.notna(df.get('or_yoy')):       parts.append(f"营收同比{df['or_yoy']:+.1f}%")
    return ' / '.join(parts)


def _build_card(rec: dict, fina_line: str, tags: list | None = None) -> str:
    """Render the entity card. Keep ≤200 tokens for bge-m3 efficiency."""
    aliases = rec.get('aliases') or []
    parts = [
        f"{rec['ts_code']} {rec.get('name','')} ({rec.get('symbol','')})",
    ]
    if rec.get('com_name'):
        parts.append(f"公司全称: {rec['com_name']}")
    if aliases:
        parts.append("别名: " + ', '.join(aliases[:8]))
    sector_bits = []
    if rec.get('sw_l1_name'): sector_bits.append(rec['sw_l1_name'])
    if rec.get('sw_l2_name'): sector_bits.append(rec['sw_l2_name'])
    if rec.get('industry'):   sector_bits.append(rec['industry'])
    if sector_bits:
        parts.append("行业: " + ' / '.join(dict.fromkeys(sector_bits)))
    if rec.get('area'):
        parts.append(f"地区: {rec['area']}")
    if rec.get('index_tags'):
        parts.append("指数: " + ', '.join(rec['index_tags']))
    leader_bits = []
    if rec.get('chairman'): leader_bits.append(f"董事长 {rec['chairman']}")
    if rec.get('manager'):  leader_bits.append(f"总经理 {rec['manager']}")
    if leader_bits:
        parts.append('管理层: ' + ', '.join(leader_bits))
    if fina_line:
        parts.append("最新业绩: " + fina_line)
    if tags:
        parts.append("热门概念: " + ', '.join(tags))
    return ' | '.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aliases', default=str(QA_DIR / 'aliases.json'))
    p.add_argument('--out_faiss', default=str(QA_DIR / 'entities.faiss'))
    p.add_argument('--out_meta',  default=str(QA_DIR / 'entities.parquet'))
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--no-fp16', action='store_true')
    args = p.parse_args()

    QA_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.aliases, 'r', encoding='utf-8') as f:
        aliases = json.load(f)
    print(f"[entity_idx] {len(aliases):,} stocks")

    # News-derived concept tags (Phase 2 enrichment) — pin distinctive
    # phrases like "钙钛矿" / "刀片电池" / "飞天" to the card so bge-m3
    # can bridge market-jargon queries that don't match any structured
    # entity field.
    concept_tags: dict = {}
    ct_path = QA_DIR / 'concept_tags.json'
    if ct_path.exists():
        with open(ct_path, 'r', encoding='utf-8') as f:
            concept_tags = json.load(f)
        n_with_tags = sum(1 for v in concept_tags.values() if v)
        print(f"[entity_idx] concept_tags: {n_with_tags:,} ts_codes tagged")

    rows = []
    for ts, rec in aliases.items():
        fina = _latest_fina_snippet(ts)
        tags = concept_tags.get(ts, [])
        card = _build_card(rec, fina, tags=tags)
        rows.append({'ts_code': ts, 'name': rec.get('name', ''), 'card': card})
    df = pd.DataFrame(rows)
    print(f"[entity_idx] cards built: {len(df):,}")
    print(f"[entity_idx] sample card: {df['card'].iloc[0][:200]}")

    embedder = BgeM3Embedder(fp16=not args.no_fp16, max_length=256)
    vecs = embedder.encode(df['card'].tolist(),
                            batch_size=args.batch, show_progress=True)
    print(f"[entity_idx] embeddings: {vecs.shape}  dtype={vecs.dtype}")

    # HNSW with inner product == cosine since vectors are L2-normalised.
    dim = vecs.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vecs.astype(np.float32))
    print(f"[entity_idx] FAISS HNSW ntotal={index.ntotal}")

    faiss.write_index(index, args.out_faiss)
    df.to_parquet(args.out_meta, index=False)

    fz = Path(args.out_faiss).stat().st_size / 1e6
    mz = Path(args.out_meta).stat().st_size / 1e6
    print(f"[entity_idx] wrote {args.out_faiss}  ({fz:.1f} MB)")
    print(f"[entity_idx] wrote {args.out_meta}   ({mz:.2f} MB)")


if __name__ == '__main__':
    main()
