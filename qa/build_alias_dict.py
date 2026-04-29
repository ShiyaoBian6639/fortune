"""
Build a per-stock alias dictionary used by both the news linker and the
question-time entity resolver.

Sources joined:
  stock_list.csv                   ts_code, symbol, name, exchange
  static_features/stock_company.csv     ts_code, com_name (full legal name), chairman, manager
  stock_sectors.csv                ts_code, sw_l1_name, sw_l2_name, industry
  static_features/index_member_pit.csv  ts_code, in_csi300/500/1000/sse50 (latest snapshot)

Per stock we emit:
  ts_code        : 600519.SH
  symbol         : 600519
  name           : 贵州茅台      (short market name)
  com_name       : 贵州茅台酒股份有限公司   (full legal name, optional)
  aliases        : [name, com_name, name_short, ...]
                   Includes:
                     - market name (`name`)
                     - full company name (`com_name`)
                     - "shortened" name = first-3-or-4 char prefix when meaningful
                     - the bare 6-digit symbol
                     - bare ts_code
  sw_l1_name     : 食品饮料
  sw_l2_name     : 白酒Ⅱ
  industry       : 白酒
  index_tags     : ["CSI300", "SSE50"]   (membership flags for context)

Output:  stock_data/qa/aliases.json   (~5,500 entries, ~2 MB)

Run:
    ./venv/Scripts/python -m qa.build_alias_dict
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'
QA_DIR = DATA / 'qa'


# Common suffixes/prefixes we strip to derive a "short" alias.
# Order matters — strip longest first.
LEGAL_SUFFIXES = [
    '股份有限公司', '有限责任公司', '有限公司', '集团股份',
    '集团有限公司', '集团有限责任公司', '集团公司', '集团',
    '股份',
]
COMMON_PREFIXES = [
    # Multi-char nations / regions
    '中国', '上海', '北京', '深圳',
    # Provinces (2-char)
    '广东', '广州', '江苏', '浙江', '山东', '山西', '河北', '河南',
    '湖北', '湖南', '辽宁', '吉林', '安徽', '江西', '福建', '海南',
    '四川', '云南', '贵州', '陕西', '甘肃', '青海', '台湾', '新疆',
    '西藏', '宁夏', '内蒙', '黑龙', '天津', '重庆', '香港', '澳门',
]


def _short_name(full: str) -> str | None:
    """Return a shortened name by stripping common legal/regional affixes.

    e.g. '贵州茅台酒股份有限公司' → '贵州茅台酒' or '贵州茅台'
         '中国平安保险(集团)股份有限公司' → '中国平安保险'
    """
    if not full or not isinstance(full, str):
        return None
    s = full.strip()
    # Strip parenthetical regions: (集团), (上海), (中国) ...
    s = re.sub(r'[\(（][^)）]{0,12}[\)）]', '', s)
    # Strip trailing legal suffixes
    for suf in LEGAL_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    s = s.strip()
    if len(s) >= 2 and s != full:
        return s
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(QA_DIR / 'aliases.json'))
    args = p.parse_args()

    QA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Stock list (ts_code ↔ market name)
    sl = pd.read_csv(DATA / 'stock_list.csv', encoding='utf-8-sig',
                      dtype={'symbol': str})
    sl = sl[['ts_code', 'symbol', 'name', 'exchange']]
    sl = sl.drop_duplicates(subset=['ts_code'])
    print(f"[alias] stock_list: {len(sl):,} rows")

    # 2. Company full names
    sc_p = DATA / 'static_features' / 'stock_company.csv'
    if sc_p.exists():
        sc = pd.read_csv(sc_p, encoding='utf-8-sig')
        sc = sc[['ts_code', 'com_name', 'chairman', 'manager']].drop_duplicates('ts_code')
    else:
        sc = pd.DataFrame(columns=['ts_code', 'com_name', 'chairman', 'manager'])
    print(f"[alias] stock_company: {len(sc):,} rows")

    # 3. Sectors (Shenwan L1 / L2 + industry)
    ss_p = DATA / 'stock_sectors.csv'
    if ss_p.exists():
        ss = pd.read_csv(ss_p, encoding='utf-8-sig')
        ss = ss[['ts_code', 'sw_l1_name', 'sw_l2_name', 'industry', 'area']].drop_duplicates('ts_code')
    else:
        ss = pd.DataFrame(columns=['ts_code', 'sw_l1_name', 'sw_l2_name', 'industry', 'area'])
    print(f"[alias] sectors: {len(ss):,} rows")

    # 4. Index membership (latest snapshot per stock)
    im_p = DATA / 'static_features' / 'index_member_pit.csv'
    if im_p.exists():
        im = pd.read_csv(im_p, encoding='utf-8-sig')
        im['snapshot_date'] = pd.to_datetime(im['snapshot_date'], errors='coerce')
        im = im.sort_values('snapshot_date').groupby('ts_code', as_index=False).last()
        im = im[['ts_code', 'in_csi300', 'in_csi500', 'in_csi1000', 'in_sse50']]
    else:
        im = pd.DataFrame(columns=['ts_code', 'in_csi300', 'in_csi500', 'in_csi1000', 'in_sse50'])
    print(f"[alias] index_member: {len(im):,} rows")

    df = sl.merge(sc, on='ts_code', how='left') \
            .merge(ss, on='ts_code', how='left') \
            .merge(im, on='ts_code', how='left')
    print(f"[alias] joined: {len(df):,} rows")

    # Tushare names sometimes contain stray full-width spaces ("五 粮 液") or
    # half-width spaces. Normalise so the alias matches a query without spaces.
    SPACE_RE = re.compile(r'[\s　]+')
    def _norm(s: str) -> str:
        return SPACE_RE.sub('', s or '').strip()

    out = {}
    stripped_alias = {}      # ts_code -> the stripped-prefix short form (or None)
    for _, r in df.iterrows():
        ts = r['ts_code']
        name = _norm(r['name'] or '')
        com  = _norm((r.get('com_name') or '')) if pd.notna(r.get('com_name')) else ''

        aliases = set()
        if name: aliases.add(name)
        if com:  aliases.add(com)
        short_com = _short_name(com)
        if short_com: aliases.add(short_com)
        # Add a "stripped-prefix" short form of the market name.
        # E.g. "贵州茅台" → "茅台", "中国平安" → "平安", "中国中免" → "中免".
        # Only applied when the result is ≥2 chars AND distinct from the
        # original — this catches the common "regional + brand" pattern in
        # A-share names without inventing nonsense aliases.
        sp = None
        for prefix in COMMON_PREFIXES:
            if name.startswith(prefix) and len(name) - len(prefix) >= 2:
                sp = name[len(prefix):]
                aliases.add(sp)
                break
        stripped_alias[ts] = sp
        # Bare-symbol matches catch references like "600519" in news
        sym = str(r['symbol']) if pd.notna(r['symbol']) else ''
        if sym: aliases.add(sym)

        idx_tags = []
        for col, tag in [('in_csi300','CSI300'), ('in_csi500','CSI500'),
                          ('in_csi1000','CSI1000'), ('in_sse50','SSE50')]:
            if pd.notna(r.get(col)) and float(r[col]) == 1.0:
                idx_tags.append(tag)

        out[ts] = {
            'ts_code':    ts,
            'symbol':     sym,
            'name':       name,
            'com_name':   com,
            'aliases':    sorted(a for a in aliases if a and len(a) >= 2),
            'sw_l1_name': r.get('sw_l1_name', '') if pd.notna(r.get('sw_l1_name')) else '',
            'sw_l2_name': r.get('sw_l2_name', '') if pd.notna(r.get('sw_l2_name')) else '',
            'industry':   r.get('industry', '')   if pd.notna(r.get('industry'))   else '',
            'area':       r.get('area', '')       if pd.notna(r.get('area'))       else '',
            'index_tags': idx_tags,
            'chairman':   r.get('chairman', '')   if pd.notna(r.get('chairman'))   else '',
            'manager':    r.get('manager', '')    if pd.notna(r.get('manager'))    else '',
        }

    # Drop over-ambiguous stripped-prefix aliases. The regional-prefix
    # stripper produces generic 2-char industry words (e.g. "深圳能源" /
    # "甘肃能源" / "湖北能源" all collapse to "能源"; "浙江新能" / "江苏新能"
    # both collapse to "新能") that then false-match unrelated queries like
    # "新能源车板块". A stripped-prefix alias is only useful when it
    # uniquely identifies its stock. If the same stripped form is the
    # short-name of any other stock, drop it from EVERY stock's alias
    # list — it's no longer a discriminating identifier.
    #
    # We restrict the rule to stripped-prefix aliases (sp) on purpose:
    # full company names and com_names are kept even when they collide,
    # since collisions there reflect genuine name overlap.
    from collections import Counter
    sp_counts = Counter(sp for sp in stripped_alias.values() if sp)
    sp_blacklist = {sp for sp, n in sp_counts.items() if n >= 2}
    if sp_blacklist:
        n_dropped = 0
        for ts, v in out.items():
            sp = stripped_alias.get(ts)
            if sp and sp in sp_blacklist and sp in v['aliases']:
                v['aliases'] = [a for a in v['aliases'] if a != sp]
                n_dropped += 1
        print(f"[alias] dropped {len(sp_blacklist)} ambiguous stripped-prefix "
              f"aliases: {sorted(sp_blacklist)[:20]}{'...' if len(sp_blacklist)>20 else ''}")
        print(f"[alias] removed {n_dropped} alias entries total")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print()
    print(f"[alias] wrote {out_p}  ({out_p.stat().st_size / 1e6:.2f} MB)")
    print(f"[alias] entries          : {len(out):,}")
    avg_aliases = sum(len(v['aliases']) for v in out.values()) / max(len(out), 1)
    print(f"[alias] avg aliases/stock: {avg_aliases:.2f}")
    print()
    # Spot-check three high-profile stocks
    for ts in ('600519.SH', '300750.SZ', '601318.SH'):
        if ts in out:
            v = out[ts]
            print(f"  {ts}: name='{v['name']}'  com='{v['com_name']}'")
            print(f"      aliases={v['aliases']}  index_tags={v['index_tags']}")


if __name__ == '__main__':
    main()
