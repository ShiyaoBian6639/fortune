"""
Retriever: maps user questions to a small set of relevant artefacts.

For the MVP we DON'T use semantic search — just:
  1. Resolve query to ts_code(s) via the alias dict (reverse lookup)
  2. Filter the linked-news parquet by ts_code
  3. Return the top-N most-recent articles per stock

Phase 2 will add bge-m3 dense retrieval within the ts_code filter so the
"top-N" ranking reflects relevance to the specific question, not just recency.

API
---
    r = Retriever('stock_data/qa/aliases.json',
                   'stock_data/qa/news_linked.parquet')
    hits = r.search("茅台最近有什么新闻", top_k=8)
    # hits = {'ts_codes': ['600519.SH'],
    #         'articles': [{'ts_code', 'date', 'title', 'content_snippet', 'source'}, ...]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SYMBOL_RE = re.compile(r'(?<![0-9])([036]\d{5})(?![0-9])')
FULL_TS_RE = re.compile(r'\b([036]\d{5})\.(SH|SZ)\b', re.I)


class Retriever:
    def __init__(self, aliases_json: str | Path,
                 news_linked_parquet: str | Path,
                 entity_index: str | Path | None = None,
                 entity_meta:  str | Path | None = None,
                 news_index:   str | Path | None = None,
                 news_meta:    str | Path | None = None,
                 semantic_top_k: int = 3,
                 semantic_min_score: float = 0.40,
                 news_semantic_min_score: float = 0.50):
        with open(aliases_json, 'r', encoding='utf-8') as f:
            self._aliases = json.load(f)        # ts_code → entry
        # Reverse maps for entity resolution
        self._name_to_ts: Dict[str, List[str]] = {}
        self._sym_to_ts:  Dict[str, str]       = {}
        for ts, v in self._aliases.items():
            self._sym_to_ts[v['symbol']] = ts
            for a in v['aliases']:
                if not a or a.isdigit() or len(a) < 2:
                    continue
                self._name_to_ts.setdefault(a, []).append(ts)
        # News
        print(f"[retriever] loading {news_linked_parquet} ...", flush=True)
        df = pd.read_parquet(news_linked_parquet)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime']).sort_values('datetime', ascending=False)
        # Explode list[str] of ts_codes for efficient per-ts lookup
        flat = df.explode('ts_codes_pred').rename(columns={'ts_codes_pred':'ts_code'})
        flat = flat.dropna(subset=['ts_code']).reset_index(drop=True)
        self._news_by_code = flat
        # Per-ts_code index for O(1) lookup
        self._news_by_code_groups = flat.groupby('ts_code', sort=False)
        # Prominence prior: log(news_count) per ts_code. Used by the
        # semantic fallback to break dense ties — 龙头 stocks have orders
        # of magnitude more press than no-name peers in the same sector.
        nc = flat['ts_code'].value_counts()
        import numpy as _np
        self._news_count = nc.to_dict()
        self._news_count_max_log = float(_np.log1p(nc.iloc[0])) if len(nc) else 1.0
        print(f"[retriever] {len(df):,} linked articles  "
              f"covering {flat['ts_code'].nunique():,} ts_codes",
              flush=True)

        # Optional dense semantic fallback (Phase 2)
        self._entity_index = None
        self._entity_meta  = None
        self._embedder     = None
        self._semantic_top_k    = semantic_top_k
        self._semantic_min_score = semantic_min_score
        if entity_index and Path(entity_index).exists() \
           and entity_meta and Path(entity_meta).exists():
            try:
                import faiss
                self._entity_index = faiss.read_index(str(entity_index))
                self._entity_meta  = pd.read_parquet(entity_meta)
                print(f"[retriever] entity index loaded "
                      f"({self._entity_index.ntotal:,} vectors)", flush=True)
            except Exception as e:
                print(f"[retriever] could not load entity index: {e}",
                      flush=True)
                self._entity_index = None
                self._entity_meta  = None

        # News semantic index — lazy-loaded on first use because it's
        # 8.5 GB. Path stored at init; index read on first call.
        self._news_index_path  = Path(news_index) if news_index else None
        self._news_meta_path   = Path(news_meta)  if news_meta  else None
        self._news_index       = None
        self._news_meta        = None
        self._news_semantic_min_score = news_semantic_min_score
        if self._news_index_path and not self._news_index_path.exists():
            print(f"[retriever] news index path missing: {self._news_index_path}",
                  flush=True)
            self._news_index_path = None
        if self._news_meta_path  and not self._news_meta_path.exists():
            print(f"[retriever] news meta path missing: {self._news_meta_path}",
                  flush=True)
            self._news_meta_path = None
        if self._news_index_path and self._news_meta_path:
            print(f"[retriever] news index path registered "
                  f"(lazy-loaded on first use)", flush=True)

    # ─── Entity resolution ─────────────────────────────────────────────────
    def resolve_entities(self, query: str) -> List[str]:
        """Extract ts_code(s) mentioned in the query.

        Order:
          1. Full ts_code form (600519.SH) — exact, highest priority
          2. Bare 6-digit symbol → resolve via _sym_to_ts
          3. Stock name / alias substring match (longest match wins)
        """
        out: List[str] = []
        seen = set()

        # 1. Full ts_codes
        for m in FULL_TS_RE.finditer(query):
            ts = f"{m.group(1)}.{m.group(2).upper()}"
            if ts in self._aliases and ts not in seen:
                seen.add(ts); out.append(ts)

        # 2. Bare 6-digit symbols
        for m in SYMBOL_RE.finditer(query):
            sym = m.group(1)
            ts = self._sym_to_ts.get(sym)
            if ts and ts not in seen:
                seen.add(ts); out.append(ts)

        # 3. Name / alias substring match. Iterate longest-first so that
        #    once we match a long alias we mask out its span — that way we
        #    won't re-match its substrings (e.g. "中国" inside "中国平安").
        #    A query like "比亚迪和宁德时代谁更强" should resolve BOTH 002594.SZ
        #    and 300750.SZ, not stop at the first hit.
        masked = list(query)
        sorted_aliases = sorted(self._name_to_ts.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            current = ''.join(masked)
            idx = current.find(alias)
            if idx >= 0:
                # Heuristic: ignore aliases that look generic — single-stock
                # names and explicit symbols are fine, but multi-mapped short
                # aliases (e.g. "中国" → 30 stocks) cause noise. Cap at 3
                # ts_codes per alias to keep precision.
                ts_list = self._name_to_ts[alias][:3]
                for ts in ts_list:
                    if ts not in seen:
                        seen.add(ts); out.append(ts)
                # Mask out the alias text so substrings don't re-match
                for i in range(idx, idx + len(alias)):
                    masked[i] = '\x00'
        return out

    # ─── Semantic fallback (Phase 2) ───────────────────────────────────────
    def _ensure_embedder(self):
        """Lazy-load bge-m3 only when the alias path fails. Pinned to CPU
        because the 4070 Super has only 12 GB VRAM — Qwen 4-bit alone
        uses ~10 GB and adding bge-m3 (~2.5 GB) on the same device pushes
        total over the limit, forcing CUDA to spill to system RAM and
        slowing every subsequent query 10–100×.

        CPU embed costs ~200–400 ms per single query (vs ~50 ms on GPU)
        which is well below the Qwen generation budget — net no harm.
        Override via ``QA_EMBED_DEVICE=cuda`` env var if you have ≥18 GB
        VRAM (e.g. 5090) and want the speed.
        """
        if self._embedder is None and self._entity_index is not None:
            import os
            from qa.rag.embedder import BgeM3Embedder
            device = os.environ.get('QA_EMBED_DEVICE', 'cpu').lower()
            self._embedder = BgeM3Embedder(
                max_length=256, device=device, fp16=(device == 'cuda'),
            )
        return self._embedder

    # Query-intent words that describe what the user wants, not which
    # entity they want. Leaving them in causes bge-m3 to match literal
    # strings ("龙头" → "龙头股份" textile co.) instead of the sector.
    _INTENT_STOP = re.compile(
        r'(板块|龙头股票|龙头股|龙头|代表公司|代表企业|代表股|龙头企业|'
        r'股票|股价|股份|公司|企业|个股|概念股|概念|行业|赛道|领涨|领跌|'
        r'最近|今天|近期|最新|怎么样|如何|是谁|是哪些|有哪些|有什么|'
        r'业绩|新闻|对比|比较|强|弱|好|差)'
    )

    # Routing keywords: detect whether the query is asking about *events
    # in the news* (news-flavored) or about *which stocks fit a category*
    # (entity-flavored). Used to pick which semantic index to consult
    # first when alias resolution fails.
    _NEWS_KEYWORDS = re.compile(
        r'(新闻|消息|动态|事件|进展|公告|政策|监管|调控|加息|降息|'
        r'利好|利空|影响|冲击|风波|突发|曝光|爆雷|暴涨|暴跌|'
        r'涨停|跌停|解禁|增持|减持|回购|分红|业绩预告|快报|'
        r'走势|行情|数据|报告|发布)'
    )
    _ENTITY_KEYWORDS = re.compile(
        r'(龙头|板块|行业|概念|赛道|个股|标的|是谁|是哪|代表|'
        r'排名|前十|top|前几|领头|主流|核心|主要)',
        re.IGNORECASE,
    )

    def _query_flavor(self, query: str) -> str:
        """Return 'news' | 'entity' | 'neutral'.

        - 'news'   : at least one news keyword AND no entity keyword
        - 'entity' : at least one entity keyword AND no news keyword
        - 'neutral': both or neither (default to current order)
        """
        has_news   = bool(self._NEWS_KEYWORDS.search(query))
        has_entity = bool(self._ENTITY_KEYWORDS.search(query))
        if has_news and not has_entity: return 'news'
        if has_entity and not has_news: return 'entity'
        return 'neutral'

    def _normalise_for_embed(self, query: str) -> str:
        s = self._INTENT_STOP.sub(' ', query)
        s = re.sub(r'[ ，,。.？?！!、；;：:\s]+', ' ', s).strip()
        return s or query   # never embed an empty string

    def semantic_resolve(self, query: str) -> List[str]:
        """Dense top-K from the entity index, then rerank by news prominence.

        Pure cosine on entity cards picks up token-level matches (e.g.
        "新能源" in any com_name, "龙头" matching "龙头股份") rather
        than industry leaders. We retrieve a wider top-K, then add a
        log-news-count prior so well-covered names (龙头 stocks have
        10×–1000× more press) bubble up.

        Final score:  cos + 0.10 · (log1p(news_count) / log1p(max_count))

        Cosine still dominates — the prior is a tie-breaker, not a
        replacement.
        """
        if self._entity_index is None or self._entity_meta is None:
            return []
        emb = self._ensure_embedder()
        if emb is None:
            return []
        import numpy as np
        q_text = self._normalise_for_embed(query)
        q = emb.encode([q_text], batch_size=1)
        # Wide initial pool — 龙头 stocks often have weaker raw cosine
        # than literal-name matches and need the prominence prior to
        # surface. 50 candidates ≈ 1 % of universe, cheap to score.
        pool_k = max(self._semantic_top_k * 16, 50)
        scores, idx = self._entity_index.search(q.astype(np.float32),
                                                 k=pool_k)
        cands: List[tuple] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0 or s < self._semantic_min_score:
                continue
            ts = self._entity_meta.iloc[i]['ts_code']
            news_n = self._news_count.get(ts, 0)
            prior = float(np.log1p(news_n) / max(self._news_count_max_log, 1e-9))
            # Prior weight 0.25: enough to flip a 0.55-cos noname under
            # a 0.45-cos 龙头 (which has 100×–1000× more news coverage).
            final = float(s) + 0.25 * prior
            cands.append((ts, final, float(s), news_n))
        cands.sort(key=lambda x: -x[1])
        return [ts for ts, _, _, _ in cands[: self._semantic_top_k]]

    def _ensure_news_index(self):
        """Lazy-load the 8.5 GB news.faiss + meta only on first use."""
        if self._news_index is not None:
            return
        if not (self._news_index_path and self._news_meta_path):
            return
        import faiss
        print(f"[retriever] loading news index "
              f"{self._news_index_path} (8+ GB)...", flush=True)
        self._news_index = faiss.read_index(str(self._news_index_path))
        self._news_meta  = pd.read_parquet(self._news_meta_path)
        print(f"[retriever] news index loaded "
              f"({self._news_index.ntotal:,} vectors)", flush=True)

    def semantic_news_search(self, query: str, top_k: int = 8,
                              ts_filter: List[str] | None = None) -> List[dict]:
        """Free-text article search over the news FAISS index.

        Used as a third fallback when neither alias nor entity-card
        semantic resolution finds a stock — typical for meta queries
        ("美联储加息对A股的影响", "光伏行业最近的政策利好") whose answer
        lives in articles, not in a single ticker.

        ``ts_filter`` (optional) restricts results to articles tagged
        with at least one of the given ts_codes via news_linked.parquet.
        """
        if self._news_index is None:
            self._ensure_news_index()
        if self._news_index is None:
            return []
        emb = self._ensure_embedder()
        if emb is None:
            return []
        import numpy as np
        q_text = self._normalise_for_embed(query)
        q = emb.encode([q_text], batch_size=1)
        # Pull a wider pool, filter, then trim.
        pool_k = top_k * (8 if ts_filter else 3)
        scores, idx = self._news_index.search(q.astype(np.float32), k=pool_k)
        hits: List[dict] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0 or s < self._news_semantic_min_score:
                continue
            row = self._news_meta.iloc[i]
            hits.append({
                'idx':      int(i),
                'score':    float(s),
                'datetime': row.get('datetime'),
                'title':    str(row.get('title') or ''),
                'source':   str(row.get('source') or ''),
                'content':  str(row.get('content_snippet') or ''),
                'content_hash': row.get('content_hash'),
            })
        # Attach ts_codes by cross-referencing news_linked.parquet via
        # content_hash (same key used to build news_meta). Articles in
        # news.faiss but absent from news_linked.parquet are unlinked
        # (no Aho-Corasick or dense match) — those will have ts_codes=[].
        if hits and 'content_hash' in self._news_by_code.columns:
            hashes = {h['content_hash'] for h in hits if h['content_hash']}
            sub = self._news_by_code[self._news_by_code['content_hash'].isin(hashes)]
            hash_to_ts = sub.groupby('content_hash')['ts_code'].apply(list).to_dict()
            for h in hits:
                h['ts_codes'] = hash_to_ts.get(h['content_hash'], [])
        else:
            for h in hits: h['ts_codes'] = []
        if ts_filter:
            ts_set = set(ts_filter)
            hits = [h for h in hits if any(t in ts_set for t in h.get('ts_codes', []))]
        return hits[:top_k]

    @staticmethod
    def _dedupe_articles(articles: List[dict], top_k: int) -> List[dict]:
        """Drop near-duplicate cross-posts. Same story re-published by
        sina / yicai / eastmoney usually keeps the headline verbatim, so
        keying on (date, normalised-title-prefix) catches them while
        being cheap. Keeps the first-seen instance (newest, since the
        flat news frame is already sorted DESC by datetime)."""
        seen = set()
        out: List[dict] = []
        for a in articles:
            d = a.get('datetime')
            d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
            t = (a.get('title') or '').strip().lower()
            # Strip whitespace + common punctuation differences across sources
            for ch in '【】[]()（）"“”\'‘’，。!！?？:：':
                t = t.replace(ch, '')
            t = t[:32]
            key = (d_str, t)
            if key in seen:
                continue
            seen.add(key)
            out.append(a)
            if len(out) >= top_k:
                break
        return out

    # ─── News retrieval ────────────────────────────────────────────────────
    def news_for(self, ts_code: str, top_k: int = 8) -> List[dict]:
        """Most recent top_k linked-news articles for a single ts_code,
        deduplicated by (date, title-prefix) so cross-posts of the same
        story don't all stack into the context."""
        if ts_code not in self._news_by_code_groups.groups:
            return []
        # Pull a wider raw slice so dedup doesn't starve the result.
        g = self._news_by_code_groups.get_group(ts_code).head(top_k * 4)
        raw = []
        for _, r in g.iterrows():
            raw.append({
                'ts_code':  ts_code,
                'datetime': r['datetime'],
                'title':    str(r.get('title') or ''),
                'content':  str(r.get('content') or '')[:400],
                'source':   str(r.get('source') or ''),
            })
        return self._dedupe_articles(raw, top_k)

    def _articles_from_news_hits(self, news_hits: List[dict]) -> tuple[List[str], List[dict]]:
        """Convert semantic_news_search hits into (ts_codes, articles).
        Cross-post duplicates are filtered out so Qwen sees distinct
        articles (instead of the same story repeated 3× from sina /
        yicai / eastmoney)."""
        from collections import Counter
        tally = Counter(t for h in news_hits for t in h.get('ts_codes', []))
        ts_codes = [ts for ts, _ in tally.most_common(self._semantic_top_k)]
        articles: List[dict] = []
        for h in news_hits:
            articles.append({
                'ts_code':  (h['ts_codes'][0] if h.get('ts_codes') else ''),
                'datetime': h['datetime'],
                'title':    h['title'],
                'content':  h.get('content', ''),
                'source':   h['source'],
            })
        return ts_codes, self._dedupe_articles(articles, top_k=len(articles))

    # ─── Combined query interface ──────────────────────────────────────────
    def search(self, query: str, top_k: int = 8) -> dict:
        """Resolve query → (ts_codes, articles).

        Resolution order:
          1. Lexical alias resolution (always first; cheap + precise).
          2. If empty, route by query flavor:
             - 'news'   : try semantic_news_search first, fall back to
                          semantic_resolve (entity index).
             - 'entity' : try semantic_resolve first, fall back to
                          semantic_news_search.
             - 'neutral': default to entity first (legacy behavior).
        """
        ts_codes = self.resolve_entities(query)
        used_semantic = False
        used_news_semantic = False
        flavor = 'lexical'

        if not ts_codes:
            flavor = self._query_flavor(query)
            if flavor == 'news':
                # Prefer the news index — query is about an event.
                news_hits = self.semantic_news_search(query, top_k=top_k)
                if news_hits:
                    ts_codes, news_articles = self._articles_from_news_hits(news_hits)
                    used_news_semantic = True
                else:
                    sem = self.semantic_resolve(query)
                    if sem:
                        ts_codes = sem
                        used_semantic = True
            else:
                # 'entity' or 'neutral' → entity index first.
                sem = self.semantic_resolve(query)
                if sem:
                    ts_codes = sem
                    used_semantic = True
                else:
                    news_hits = self.semantic_news_search(query, top_k=top_k)
                    if news_hits:
                        ts_codes, news_articles = self._articles_from_news_hits(news_hits)
                        used_news_semantic = True

        # Build the articles payload. If we used the news path, those
        # articles already carry the matched titles; otherwise pull
        # recent news per ts_code from the linked corpus.
        articles: List[dict] = []
        if used_news_semantic:
            articles = news_articles  # noqa: F821 — assigned in branch above
        else:
            for ts in ts_codes:
                articles.extend(self.news_for(ts, top_k=top_k))

        return {'ts_codes': ts_codes, 'articles': articles,
                'used_semantic': used_semantic,
                'used_news_semantic': used_news_semantic,
                'flavor': flavor}


def _self_test():
    r = Retriever('stock_data/qa/aliases.json',
                   'stock_data/qa/news_linked.parquet')
    for q in ("贵州茅台最近的新闻是什么？",
              "300750.SZ 业绩怎么样",
              "比亚迪和宁德时代谁的毛利率更高",
              "600519 最近一个月业绩如何"):
        print(f"\n=== {q} ===")
        out = r.search(q, top_k=3)
        print(f"  ts_codes resolved: {out['ts_codes']}")
        for a in out['articles']:
            print(f"  - {a['ts_code']}  {a['datetime'].date()}  "
                  f"[{a['source']}] {a['title'][:50]}")


if __name__ == '__main__':
    _self_test()
