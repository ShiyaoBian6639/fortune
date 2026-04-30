"""
Train a logistic-regression reranker over (article, candidate ts_code)
pairs to lift news-corpus → ts_code coverage beyond the Aho-Corasick floor.

Pipeline
--------
1. Positives: ``news_corpus_dedup_codes.parquet`` (~70 K regex-confirmed
   article ↔ ts_code pairs, treated as ground truth).
2. For each positive article, run dense top-K against the entity index.
   The true ts_code is the positive; other dense candidates that are NOT
   in the article's true ts_codes are used as hard negatives.
3. Per (article, candidate ts_code) features:
     - lex_count_title:    Aho-Corasick hits in title
     - lex_count_content:  AC hits in content (capped at 3)
     - dense_cos:          bge-m3 cosine
     - same_sw_l1:         sector co-occurrence prior with positives
     - title_len, content_len: log-scaled
4. Fit sklearn LogisticRegression(C=1.0). Save weights + feature schema
   to ``stock_data/qa/linker_reranker.pkl``.

The trained model is consumed by ``qa.linker.predict --use_dense`` to
score dense candidates that the lexical pass missed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from qa.linker.ahocorasick_matcher import build_matcher, match_article
from qa.linker.dense_matcher import DenseEntityMatcher

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'stock_data'
QA_DIR = DATA / 'qa'


def _ac_hits(matcher_tuple, text: str) -> int:
    """Count AC hits across all aliases that map to a given ts_code."""
    matcher, _sym_to_ts = matcher_tuple
    hits = 0
    for _, _ in matcher.iter(text):
        hits += 1
    return hits


def _features(article_text: str, title: str,
              candidate_ts: str, dense_cos: float,
              ts_aliases: dict) -> List[float]:
    aliases = ts_aliases.get(candidate_ts, [])
    lex_title   = sum(1 for a in aliases if a and a in title)
    lex_content = sum(1 for a in aliases if a and a in article_text)
    return [
        float(min(lex_title,   3)),
        float(min(lex_content, 5)),
        float(dense_cos),
        float(np.log1p(len(article_text))),
        float(np.log1p(len(title))),
    ]


FEATURE_NAMES = [
    'lex_title_capped3', 'lex_content_capped5',
    'dense_cos', 'log1p_len_content', 'log1p_len_title',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--positives',  default=str(DATA / 'news_corpus_dedup_codes.parquet'))
    p.add_argument('--aliases',    default=str(QA_DIR / 'aliases.json'))
    p.add_argument('--entity_faiss', default=str(QA_DIR / 'entities.faiss'))
    p.add_argument('--entity_meta',  default=str(QA_DIR / 'entities.parquet'))
    p.add_argument('--out',         default=str(QA_DIR / 'linker_reranker.pkl'))
    p.add_argument('--max_pos',     type=int, default=10_000,
                   help='Cap positives sampled for training (speed).')
    p.add_argument('--top_k',       type=int, default=20,
                   help='Dense candidates per article.')
    p.add_argument('--max_content', type=int, default=512)
    p.add_argument('--batch',       type=int, default=64)
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression

    # ── 1. Load data ─────────────────────────────────────────────────────
    print(f"[rerank] loading positives {args.positives} ...")
    pos = pd.read_parquet(args.positives)
    pos = pos.dropna(subset=['title', 'content'])
    pos = pos[pos['ts_codes'].str.len() > 0]
    if len(pos) > args.max_pos:
        pos = pos.sample(args.max_pos, random_state=42).reset_index(drop=True)
    print(f"[rerank] positives sampled: {len(pos):,}")

    with open(args.aliases, 'r', encoding='utf-8') as f:
        alias_dict = json.load(f)
    ts_aliases = {ts: v.get('aliases', []) for ts, v in alias_dict.items()}

    # ── 2. Dense candidates ─────────────────────────────────────────────
    print(f"[rerank] embedding {len(pos):,} articles ...")
    dm = DenseEntityMatcher(args.entity_faiss, args.entity_meta)
    article_texts = (pos['title'].fillna('') + '。 '
                     + pos['content'].fillna('').str[:args.max_content]).tolist()
    t0 = time.time()
    cands_per_art: List[list] = []
    for i in range(0, len(article_texts), args.batch):
        batch = article_texts[i: i + args.batch]
        cands_per_art.extend(dm.search(batch, top_k=args.top_k,
                                        batch_size=args.batch))
        if i and i % (args.batch * 50) == 0:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  [{i:,}/{len(article_texts):,}]  "
                  f"{rate:.0f} art/s", flush=True)
    print(f"[rerank] dense candidate retrieval done "
          f"({len(cands_per_art):,} articles)")

    # ── 3. Build feature matrix ─────────────────────────────────────────
    X: List[List[float]] = []
    y: List[int]         = []
    for art_text, title, true_codes, cands in zip(
        pos['content'].fillna('').str[:args.max_content].tolist(),
        pos['title'].fillna('').tolist(),
        pos['ts_codes'].tolist(),
        cands_per_art,
    ):
        true_set = set(true_codes)
        # Reranker scope: only candidates the dense layer actually
        # surfaced. Positives outside dense top-K are out-of-scope
        # (the reranker can't help recover them) and must NOT be added
        # — otherwise we contaminate the cos distribution with cos=0
        # positives and the model learns the wrong sign on cosine.
        for ts, cos in cands:
            if ts in true_set:
                X.append(_features(art_text, title, ts, cos, ts_aliases))
                y.append(1)
            else:
                X.append(_features(art_text, title, ts, cos, ts_aliases))
                y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[rerank] features: X={X.shape}  pos={int(y.sum())}  neg={int((1-y).sum())}")

    # ── 4. Train + report ───────────────────────────────────────────────
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    cut = int(n * 0.85)
    tr, te = perm[:cut], perm[cut:]

    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    clf.fit(X[tr], y[tr])
    train_acc = clf.score(X[tr], y[tr])
    test_acc  = clf.score(X[te], y[te])
    coefs = dict(zip(FEATURE_NAMES, clf.coef_[0].round(3).tolist()))
    print(f"[rerank] train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")
    print(f"[rerank] coef: {coefs}  intercept={clf.intercept_[0]:.3f}")

    # ── 5. Save ─────────────────────────────────────────────────────────
    bundle = {
        'model': clf,
        'feature_names': FEATURE_NAMES,
        'metrics': {'train_acc': train_acc, 'test_acc': test_acc,
                    'coef': coefs,
                    'intercept': float(clf.intercept_[0])},
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[rerank] saved → {out_p}")


if __name__ == '__main__':
    main()
