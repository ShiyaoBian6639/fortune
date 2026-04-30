"""
Mine distinctive concept phrases per ts_code from its linked news titles.

Why: bge-m3 over the structured entity cards can't bridge market-jargon
concepts that don't appear in any card's vocabulary. A query like
"钙钛矿电池标的" never matches 协鑫集成 / 京山轻机 because nothing in
their cards mentions 钙钛矿 — but their *news titles* do, repeatedly.
We harvest those repeated, stock-specific phrases and pin them to the
entity card so the dense retriever has a hook.

Algorithm: per-ts_code TF-IDF over 2-4 char Chinese n-grams from news
titles. We pick phrases that are:
  - frequent locally (≥ 2 hits across this ts_code's titles)
  - rare globally (document frequency < 5 % of ts_codes)
  - not in the financial-Chinese stopword blacklist
  - not subsumed by an already-picked longer phrase

Output: stock_data/qa/concept_tags.json  ts_code → [up to 6 phrases]

Run:
    ./venv/Scripts/python -m qa.build_concept_tags
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'
QA_DIR = DATA / 'qa'

# Generic financial-Chinese phrases that always appear and tell us nothing
# about a stock's actual business. Strict on the longer ones (公司公告 →
# always seen) and looser on shorter (some 2-grams like 公司 are caught
# by the global-frequency filter alone).
_BLACKLIST = set("""
公司 业绩 报告 公告 股份 有限 集团 科技 控股 实业 股东 董事 监事 高管
薪酬 议案 提案 决议 表决 审议 通过 召开 委员 委员会 大会 投票 选举
预案 安排 利润 分配 派息 派发 决算 财务 内部 控制 自我 评价 募集
募集资金 投资 投资者 关系 互动 平台 易主 换手 换股 流通 公告 关注 警示
重大 事项 进展 风险 提示 主要 主营 经营 运营 工作 管理 高级 职务
辞任 辞职 任命 聘任 续聘 解聘 接任 接班 增持 减持 回购 分红 派发
涨停 跌停 收涨 收跌 涨幅 跌幅 上涨 下跌 上行 下行 走低 走高
高位 低位 区间 周期 季度 年度 月度 日报 周报 月报 季报 年报
披露 发布 公布 通告 公开 关于 编制 出具 鉴定 复核 审计 中期
人民币 亿元 万元 千元 百分点 比例 同比 环比 期间 截至 报告期
公司控股 控股股东 实际控制人 间接持有 直接持有 一致行动人
有限责任公司 股份有限公司 集团股份 集团有限 子公司 子集团
2020 2021 2022 2023 2024 2025 2026 2027
恒生 生指 生指数 港股 港股收盘 港股开盘 港股午评 恒生科技 科技指 科技指数
南向资金 北向资金 亿港元 港交所 港股通 中概股 中概 闪崩
A股 美股 日股 欧股 大盘 沪指 深成指 创业板指 科创50
板块 概念股 异动 拉升 跳水 翻红 翻绿 走强 走弱 集体 个股
午评 收评 早评 盘中 盘前 盘后 早间 晚间 当日 全天
""".split())

_GENERIC_VERBS = ('成立', '认购', '签订', '协议', '合同', '完成', '取得', '获得')

# Punctuation / whitespace to strip before n-gram extraction.
_STRIP_RE = re.compile(r'[\s。，！？：；、|/\\,.!?:;()\[\]【】《》"""\'\'\'－—\-—…·*]+')


def _ngrams(text: str, n_min: int = 2, n_max: int = 4) -> list:
    """Yield n-grams (Chinese chars + alnum) from a single title."""
    if not isinstance(text, str): return []
    cleaned = _STRIP_RE.sub('', text)
    out = []
    for n in range(n_min, n_max + 1):
        for i in range(len(cleaned) - n + 1):
            g = cleaned[i: i + n]
            if g.isdigit(): continue
            if any(c.isspace() for c in g): continue
            out.append(g)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--news_linked', default=str(QA_DIR / 'news_linked.parquet'))
    p.add_argument('--out',         default=str(QA_DIR / 'concept_tags.json'))
    p.add_argument('--max_titles',  type=int, default=200,
                   help='Cap on titles per ts_code (most-recent first).')
    p.add_argument('--top_per_ts',  type=int, default=6,
                   help='How many concept tags to retain per ts_code.')
    p.add_argument('--max_df_frac', type=float, default=0.02,
                   help='Drop n-grams above this doc-frequency fraction.')
    p.add_argument('--min_local',   type=int, default=3,
                   help='Min times the n-gram must appear in this stock '
                        's titles to be considered.')
    args = p.parse_args()

    QA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[concept_tags] loading {args.news_linked} ...")
    df = pd.read_parquet(args.news_linked, columns=['datetime', 'title', 'ts_codes_pred'])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime', ascending=False)
    flat = df.explode('ts_codes_pred').rename(columns={'ts_codes_pred': 'ts_code'})
    flat = flat.dropna(subset=['ts_code', 'title'])
    print(f"[concept_tags] {len(flat):,} (ts, title) pairs across "
          f"{flat['ts_code'].nunique():,} ts_codes")

    # Pass 1: collect distinct n-grams per ts_code (capped to N most-recent
    # titles), and compute global document frequency.
    t0 = time.time()
    df_counter: Counter = Counter()
    ts_local_tf: dict = {}  # ts_code → Counter of n-grams (with multiplicity)
    for ts_code, g in flat.groupby('ts_code', sort=False):
        titles = g['title'].head(args.max_titles).tolist()
        local = Counter()
        seen = set()
        for t in titles:
            for ng in _ngrams(t):
                local[ng] += 1
                seen.add(ng)
        ts_local_tf[ts_code] = local
        for ng in seen:
            df_counter[ng] += 1
    print(f"[concept_tags] pass1 done  ({time.time()-t0:.1f}s)  "
          f"distinct n-grams: {len(df_counter):,}")

    N = len(ts_local_tf)
    df_max = int(N * args.max_df_frac)
    print(f"[concept_tags] cutoff: drop n-grams in > {df_max} ts_codes")

    # Pass 2: TF-IDF score per (ts_code, n-gram), keep distinctive ones.
    tags_by_ts: dict = {}
    n_total_tags = 0
    for ts_code, tf in ts_local_tf.items():
        scored = []
        for ng, c in tf.items():
            df_n = df_counter[ng]
            if c < args.min_local: continue
            if df_n > df_max: continue
            if df_n < 2: continue          # singleton, probably typo
            if ng in _BLACKLIST: continue
            if any(v in ng for v in _GENERIC_VERBS): continue
            # Filter pure-numeric or trivial like "20年"
            if re.fullmatch(r'\d{2,}[年月日]?', ng): continue
            idf = math.log(N / df_n)
            scored.append((ng, c * idf, c, df_n))
        scored.sort(key=lambda x: -x[1])

        # De-dupe by substring containment: when both "钙钛矿" and
        # "钙钛矿电池" are candidates we want only one. Greedy pass
        # keeps the higher-scoring of any overlap pair, biased toward
        # longer (more specific) phrases when scores are close.
        picked: list = []   # parallel arrays
        picked_scores: list = []
        for ng, score, _, _ in scored:
            replace_idx = -1
            redundant = False
            for j, p_ng in enumerate(picked):
                if ng in p_ng or p_ng in ng:
                    if len(ng) > len(p_ng) and score >= 0.7 * picked_scores[j]:
                        replace_idx = j
                    redundant = True
                    break
            if replace_idx >= 0:
                picked[replace_idx]        = ng
                picked_scores[replace_idx] = score
                continue
            if redundant: continue
            picked.append(ng)
            picked_scores.append(score)
            if len(picked) >= args.top_per_ts: break
        tags_by_ts[ts_code] = picked
        n_total_tags += len(picked)

    print(f"[concept_tags] avg tags per ts: {n_total_tags / max(N,1):.2f}")

    out_p = Path(args.out)
    out_p.write_text(
        json.dumps(tags_by_ts, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"[concept_tags] wrote {out_p}  "
          f"({out_p.stat().st_size / 1e6:.2f} MB)")

    # Spot-check
    samples = ['600519.SH', '300750.SZ', '002594.SZ', '002514.SZ',
                '002459.SZ', '300274.SZ', '688981.SH']
    print("\nspot-check:")
    for ts in samples:
        if ts in tags_by_ts:
            print(f"  {ts}  -> {tags_by_ts[ts]}")


if __name__ == '__main__':
    main()
