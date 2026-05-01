"""
Build a structured Markdown context block to feed Qwen for stock Q&A.

Per ts_code we assemble (in order, capped to ~3,500 tokens total):

  1. Entity card  — name, full company, sector, industry, index tags
  2. Latest 4 quarters of fundamentals from fina_indicator/<code>.csv
     (eps, roe, roa, gross/net margin, growth)
  3. 30-day price summary from sh|sz/<code>.csv
     (close, pct_chg, high/low range, momentum)
  4. Top-N most-recent linked news from the Retriever
  5. Sentiment trend over last 30 days from news_sentiment_qwen.csv

API
---
    cb = ContextBuilder(aliases='stock_data/qa/aliases.json')
    text = cb.build_for(ts_codes=['600519.SH', '300750.SZ'],
                         articles=articles_from_retriever,
                         max_tokens=3500)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'stock_data'


def _stock_csv_path(ts_code: str) -> Path:
    code, suf = ts_code.split('.')
    sub = 'sh' if suf.upper() == 'SH' else 'sz'
    return DATA / sub / f'{code}.csv'


def _fina_csv_path(ts_code: str) -> Path:
    """fina_indicator file naming uses underscore (000001_SZ.csv), not dot."""
    code, suf = ts_code.split('.')
    return DATA / 'fina_indicator' / f'{code}_{suf.upper()}.csv'


def _approx_tokens(text: str) -> int:
    """Cheap token-count: 1 token ≈ 1.6 Chinese chars or 4 ASCII chars."""
    n_cn = sum(1 for c in text if '一' <= c <= '鿿')
    n_other = len(text) - n_cn
    return int(n_cn / 1.6 + n_other / 4)


class ContextBuilder:
    def __init__(self, aliases: str | Path):
        with open(aliases, 'r', encoding='utf-8') as f:
            self._aliases = json.load(f)
        # Lazy-load sentiment table
        sp = DATA / 'news_sentiment_qwen.csv'
        if sp.exists():
            self._sentiment = pd.read_csv(sp, encoding='utf-8-sig',
                                            parse_dates=['trade_date'])
        else:
            self._sentiment = None

    # ─── Per-section formatters ────────────────────────────────────────────
    def _fmt_entity(self, ts_code: str) -> str:
        v = self._aliases.get(ts_code)
        if not v: return f'## {ts_code}\n(no entity card)\n'
        idx = ', '.join(v.get('index_tags', [])) or '—'
        return (f"## {v['name']}（{ts_code}）\n"
                f"- 全称：{v.get('com_name','—')}\n"
                f"- 申万一级：{v.get('sw_l1_name','—')}　二级：{v.get('sw_l2_name','—')}　行业：{v.get('industry','—')}\n"
                f"- 地区：{v.get('area','—')}　董事长：{v.get('chairman','—')}　总经理：{v.get('manager','—')}\n"
                f"- 指数成分：{idx}\n")

    def _fmt_fundamentals(self, ts_code: str, n_quarters: int = 4) -> str:
        fp = _fina_csv_path(ts_code)
        if not fp.exists():
            return f'### 业绩（最近{n_quarters}季）\n(无数据)\n'
        df = pd.read_csv(fp, encoding='utf-8-sig', dtype={'ann_date': str, 'end_date': str})
        df = df.sort_values('end_date', ascending=False).head(n_quarters)
        if df.empty:
            return f'### 业绩（最近{n_quarters}季）\n(无数据)\n'
        cols_show = [
            ('end_date',           '报告期'),
            ('ann_date',           '披露日'),
            ('eps',                'EPS（元）'),
            ('roe',                'ROE %'),
            ('roa',                'ROA %'),
            ('grossprofit_margin', '毛利率 %'),
            ('netprofit_margin',   '净利率 %'),
            ('op_yoy',             '营收YoY %'),
            ('netprofit_yoy',      '归母净利YoY %'),
            ('debt_to_assets',     '资产负债率 %'),
        ]
        rows_md = ['| ' + ' | '.join(label for _, label in cols_show) + ' |',
                   '|' + '|'.join('---' for _ in cols_show) + '|']
        for _, r in df.iterrows():
            cells = []
            for c, _ in cols_show:
                v = r.get(c, '')
                if pd.isna(v):
                    cells.append('—')
                elif isinstance(v, float):
                    cells.append(f'{v:.2f}')
                else:
                    cells.append(str(v))
            rows_md.append('| ' + ' | '.join(cells) + ' |')
        return f'### 业绩（最近{n_quarters}季）\n' + '\n'.join(rows_md) + '\n'

    def _fmt_price(self, ts_code: str, n_days: int = 30) -> str:
        fp = _stock_csv_path(ts_code)
        if not fp.exists():
            return f'### 价格（近{n_days}日）\n(无数据)\n'
        df = pd.read_csv(fp, encoding='utf-8-sig',
                          usecols=['trade_date','open','high','low','close','pct_chg'],
                          dtype={'trade_date': str})
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date', ascending=False).head(n_days)
        if df.empty:
            return f'### 价格（近{n_days}日）\n(无数据)\n'
        last_close = float(df.iloc[0]['close'])
        first_close = float(df.iloc[-1]['close'])
        period_pct = (last_close / first_close - 1.0) * 100
        period_high = float(df['high'].max())
        period_low  = float(df['low'].min())
        last_date = df.iloc[0]['trade_date'].strftime('%Y-%m-%d')
        return (f"### 价格（近{n_days}日，截至 {last_date}）\n"
                f"- 收盘价：{last_close:.2f}　区间涨跌：{period_pct:+.2f}%　"
                f"区间最高：{period_high:.2f}　最低：{period_low:.2f}\n"
                f"- 最近一日 pct_chg：{float(df.iloc[0]['pct_chg']):+.2f}%\n")

    def _fmt_sentiment(self, ts_code: str, n_days: int = 30) -> str:
        if self._sentiment is None:
            return ''
        sub = self._sentiment[self._sentiment['ts_code'] == ts_code]
        if sub.empty:
            return f'### 新闻情绪（近{n_days}日）\n(无相关新闻)\n'
        cutoff = sub['trade_date'].max() - pd.Timedelta(days=n_days)
        sub = sub[sub['trade_date'] >= cutoff]
        if sub.empty:
            return f'### 新闻情绪（近{n_days}日）\n(无相关新闻)\n'
        return (f"### 新闻情绪（近{n_days}日）\n"
                f"- 文章数：{int(sub['n_articles'].sum())}　"
                f"情绪均值：{sub['sentiment_mean'].mean():+.3f}　"
                f"正面占比：{sub['sentiment_pos_share'].mean()*100:.1f}%　"
                f"负面占比：{sub['sentiment_neg_share'].mean()*100:.1f}%\n")

    def _fmt_news(self, articles: List[dict], ts_codes: List[str],
                   max_per_code: int = 5) -> str:
        if not articles:
            return '### 相关新闻\n(无)\n'

        def _render(a: dict) -> str:
            d = a['datetime'].strftime('%Y-%m-%d') \
                if hasattr(a['datetime'], 'strftime') else str(a['datetime'])[:10]
            title = (a.get('title') or '').strip() or '(无标题)'
            snip  = (a.get('content') or '').strip()[:200]
            return f"[{d}] [{a.get('source','')}] **{title}**\n   {snip}"

        out = ['### 相关新闻']
        if ts_codes:
            # Stock-specific: group by ts_code so the LLM can attribute
            # each article to its company. Use bullets (not numbered)
            # so the model picks its own numbering and starts at 1.
            for ts in ts_codes:
                ts_articles = [a for a in articles if a['ts_code'] == ts][:max_per_code]
                if not ts_articles: continue
                out.append(f'**{ts}：**')
                for a in ts_articles:
                    out.append(f"- {_render(a)}")
        else:
            # Meta query (e.g. "美联储加息对A股的影响"): no specific
            # stock — render the articles flat so Qwen can synthesize
            # an answer directly from them.
            for a in articles[:max_per_code * 2]:
                out.append(f"- {_render(a)}")
        return '\n'.join(out) + '\n'

    # ─── Forecast section ──────────────────────────────────────────────────
    def _fmt_forecast(self, ts_code: str, fp) -> str:
        """Render the modelfactory vote summary + top features for one
        ts_code. Always emits the disclaimer line. Returns '' if the
        ts_code isn't covered by the live predictions."""
        s = fp.summary(ts_code, n_days=14, max_features=5,
                        max_with_values=2)
        if not s.get('covered'):
            return ''

        v = self._aliases.get(ts_code, {})
        name = v.get('name', '')
        out = [f"### 模型预测 — 仅供参考 ({ts_code} {name})",
               f"预测日期：{s.get('trade_date','—')}"]

        n_total = s['n_models_total']
        n_up    = s['n_models_up']
        out.append(f"- 模型投票：**{n_up} / {n_total} 看多**")
        if s.get('binary_up_prob') is not None:
            out.append(f"- 二分类上涨概率（xgb+lgb+catboost 均值）: "
                       f"{100*s['binary_up_prob']:.1f}%")
        if s.get('reg_next_day_pct') is not None:
            out.append(f"- 次日收益预测（regression 均值）: "
                       f"{s['reg_next_day_pct']:+.2f}%")
        if s.get('reg_5d_pct') is not None:
            out.append(f"- 5 日累计收益预测（regression 均值）: "
                       f"{s['reg_5d_pct']:+.2f}%")

        if s.get('top_features'):
            out.append('')
            out.append('**Top 关键特征**（按 XGB regression r_close_close 重要性）：')
            for i, f in enumerate(s['top_features'], 1):
                desc = f' — {f["desc"]}' if f.get('desc') else ''
                imp  = f.get('importance')
                imp_str = f"importance {imp/1e6:.2f}M" if imp else "(TA fallback)"
                line = f"{i}. `{f['name']}`{desc}（{imp_str}）"
                if f.get('recent'):
                    tail = f['recent'][-7:]   # last week's worth
                    series = ' → '.join(f"{val:.2f}" for _, val in tail)
                    line += f" 近期: {series}"
                out.append(line)

        out.append('')
        out.append('> 基于模型预测，仅供参考，不构成投资建议。')
        return '\n'.join(out) + '\n'

    # ─── Top-level interface ───────────────────────────────────────────────
    def build_for(self, ts_codes: List[str], articles: List[dict],
                   max_tokens: int = 3500,
                   fp = None,
                   include_forecast: bool = False) -> str:
        sections = []
        for ts in ts_codes:
            sections.append(self._fmt_entity(ts))
            sections.append(self._fmt_fundamentals(ts))
            sections.append(self._fmt_price(ts))
            sections.append(self._fmt_sentiment(ts))
            if include_forecast and fp is not None:
                fc = self._fmt_forecast(ts, fp)
                if fc: sections.append(fc)
        sections.append(self._fmt_news(articles, ts_codes))

        # Greedy concat with a token-budget guard
        out, total_t = [], 0
        for s in sections:
            t = _approx_tokens(s)
            if total_t + t > max_tokens and out:
                break
            out.append(s); total_t += t
        return '\n'.join(out)


def _self_test():
    cb = ContextBuilder('stock_data/qa/aliases.json')
    from qa.rag.retriever import Retriever
    r = Retriever('stock_data/qa/aliases.json',
                   'stock_data/qa/news_linked.parquet')
    out = r.search('300750.SZ 最近一季度的业绩怎么样', top_k=3)
    ctx = cb.build_for(out['ts_codes'], out['articles'])
    print(ctx)
    print(f"\n[ctx-builder] tokens (approx): {_approx_tokens(ctx)}")


if __name__ == '__main__':
    _self_test()
