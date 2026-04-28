"""Self-contained dashboard for A-share delisting risk prediction.

Reads stock_data/delist_predictions.json (built by xgbmodel.delist_predict)
and writes dashboard/delist.html — single HTML, embedded JSON, Plotly via CDN.

Usage:
    ./venv/Scripts/python -m dashboard.build_delist
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt_pct(x, dp=2):
    if x is None:
        return "—"
    return f"{x*100:.{dp}f}%"


def fmt_num(x, dp=4):
    if x is None:
        return "—"
    return f"{x:.{dp}f}"


def metric_card(label, value, sub=None, kind="neutral"):
    sub_html = f"<div class='m-sub'>{sub}</div>" if sub else ""
    return (f"<div class='card kind-{kind}'>"
            f"<div class='m-label'>{label}</div>"
            f"<div class='m-value'>{value}</div>"
            f"{sub_html}</div>")


def build_html(d: dict) -> str:
    metrics = d.get("metrics", {})
    test = metrics.get("test", {})
    val = metrics.get("val", {})
    feature_importance = d.get("feature_importance", [])
    predictions = d.get("predictions", [])

    n_high = sum(1 for r in predictions if r["p_delist"] >= 0.30)
    n_med = sum(1 for r in predictions if 0.05 <= r["p_delist"] < 0.30)
    n_low = sum(1 for r in predictions if r["p_delist"] < 0.05)

    feat_meaning = {
        "log_close":             "对数收盘价 (越低越接近退市价)",
        "ret_30":                "近30日累计收益",
        "ret_90":                "近90日累计收益",
        "ret_180":               "近180日累计收益",
        "ret_365":               "近365日累计收益",
        "drawdown_1y":           "距1年最高价回撤 (越接近-1越深)",
        "drawdown_3y":           "距3年最高价回撤",
        "dist_from_1y_low":      "距1年最低价距离 (越小越接近底部)",
        "realized_vol_60":       "60日实现波动率 (年化)",
        "avg_range_60":          "60日日均振幅 (high-low)/close",
        "log_avg_amount_60":     "60日对数日均成交额",
        "log_avg_amount_365":    "365日对数日均成交额",
        "amount_trend":          "成交额近-远比 (负=萎缩)",
        "zero_vol_days_90":     "近90日零成交日数 (停牌)",
        "low_vol_days_90":      "近90日极低成交日数",
        "log_circ_mv":           "对数流通市值 (小盘风险更高)",
        "pb":                    "市净率",
        "pe_ttm":                "市盈率 TTM (clip ±200)",
        "turnover_60d":          "60日平均自由换手率%",
        "is_st":                 "当前 ST/*ST",
        "is_starred_st":         "当前 *ST",
        "days_in_st_total":      "历史累计 ST 天数",
        "n_st_episodes":         "历史 ST 次数",
        "days_since_last_st_end":"距最近一次摘帽天数",
        "days_listed":           "上市天数",
        "is_main":               "主板",
        "is_chinext":            "创业板",
        "is_star":               "科创板",
        "is_bse":                "北交所",
    }

    cards_html = "".join([
        metric_card("当前在册股票", f"{d['n_listed']:,}",
                    sub=f"已退市历史: {d['n_delisted_total']}"),
        metric_card("高风险 (≥30%)", f"{n_high}",
                    sub=f"占在册 {n_high/max(d['n_listed'],1)*100:.2f}%",
                    kind="down"),
        metric_card("中风险 (5-30%)", f"{n_med}", kind="neutral"),
        metric_card("低风险 (<5%)", f"{n_low}", kind="up"),
        metric_card("Test AUC (2024+)", fmt_num(test.get("auc"), 4),
                    sub=f"AP {fmt_num(test.get('ap'),4)} · base {fmt_num(test.get('base_rate'),4)}",
                    kind="up"),
        metric_card("Val AUC (2023)", fmt_num(val.get("auc"), 4),
                    sub=f"AP {fmt_num(val.get('ap'),4)}"),
    ])

    # Yearly delistings
    by_year = d.get("delistings_by_year", {})
    years = sorted(by_year.keys())
    year_table_rows = "".join(
        f"<tr><td>{y}</td><td>{by_year[y]}</td></tr>"
        for y in years if y >= "2010"
    )

    # Feature meaning side panel (used by JS to render per-stock detail)
    feat_meaning_json = json.dumps(feat_meaning, ensure_ascii=False)

    # Risk-tier color mapping
    payload = json.dumps(d, ensure_ascii=False, separators=(",", ":"), default=str)

    HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A股退市风险预测</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg:#0e1117; --panel:#161b22; --panel2:#1c232e; --border:#2a313c;
    --text:#c9d1d9; --muted:#7d8590; --accent:#58a6ff;
    --up:#da3633; --down:#2ea043; --warn:#d29922;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "PingFang SC","Microsoft YaHei",-apple-system,sans-serif;
         background:var(--bg); color:var(--text); margin:0; line-height:1.5; }}
  header {{ background:linear-gradient(180deg,#1a2330 0%,#0e1117 100%);
           padding:24px 40px; border-bottom:1px solid var(--border); }}
  header h1 {{ margin:0 0 6px; font-size:24px; }}
  header p {{ margin:0; color:var(--muted); font-size:13px; }}
  nav {{ position:sticky; top:0; background:var(--panel); padding:8px 40px;
         border-bottom:1px solid var(--border); z-index:10; }}
  nav a {{ color:var(--muted); padding:6px 12px; margin-right:4px;
          text-decoration:none; font-size:13px; border-radius:4px; }}
  nav a:hover {{ background:var(--panel2); color:var(--text); }}
  section {{ padding:24px 40px; border-bottom:1px solid var(--border); }}
  section h2 {{ margin:0 0 4px; font-size:18px; font-weight:600; }}
  section .subtitle {{ color:var(--muted); font-size:12px; margin:0 0 14px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
           gap:10px; margin-bottom:20px; }}
  .card {{ background:var(--panel); border:1px solid var(--border);
           border-radius:6px; padding:14px;
           border-left:3px solid var(--accent); }}
  .card.kind-up {{ border-left-color:var(--down); }}
  .card.kind-down {{ border-left-color:var(--up); }}
  .m-label {{ color:var(--muted); font-size:11px; text-transform:uppercase;
             letter-spacing:0.5px; }}
  .m-value {{ font-size:22px; font-weight:600; margin:4px 0; }}
  .m-sub {{ color:var(--muted); font-size:11px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px;
          background:var(--panel); }}
  th, td {{ padding:6px 10px; border-bottom:1px solid var(--border); text-align:right; }}
  th {{ background:var(--panel2); color:var(--muted); font-weight:500;
        font-size:11px; text-transform:uppercase; letter-spacing:0.5px;
        text-align:right; cursor:pointer; user-select:none; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  tr:hover {{ background:var(--panel2); }}
  tr.selected {{ background:rgba(88,166,255,.18); }}
  .pos {{ color:var(--down); }}
  .neg {{ color:var(--up); }}
  .pill {{ display:inline-block; padding:1px 8px; border-radius:3px;
          font-size:11px; font-weight:500; }}
  .pill.high {{ background:rgba(218,54,51,.20); color:#f87171; }}
  .pill.med  {{ background:rgba(218,153,34,.20); color:#fbbf24; }}
  .pill.low  {{ background:rgba(46,160,67,.20); color:#86efac; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .grid-2 > div.left {{ flex: 2; }}
  @media (max-width:1100px) {{ .grid-2 {{ grid-template-columns:1fr; }} }}
  .table-wrap {{ overflow-y:auto; border:1px solid var(--border);
                border-radius:6px; }}
  .controls {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center;
              margin-bottom:10px; font-size:13px; color:var(--muted); }}
  .controls input, .controls select {{ background:var(--panel2);
              color:var(--text); border:1px solid var(--border);
              padding:5px 8px; border-radius:4px; font-size:13px; }}
  .stockdetail {{ background:var(--panel); border:1px solid var(--border);
                 border-radius:6px; padding:16px; min-height:340px; }}
  .stockdetail h3 {{ margin:0 0 4px; font-size:16px; color:var(--text); }}
  .stockdetail .head-line {{ color:var(--muted); font-size:12px; margin:0 0 12px; }}
  .feat-row {{ display:grid; grid-template-columns:24px 1fr 90px 110px;
               gap:8px; padding:5px 0; border-bottom:1px dotted var(--border);
               font-size:12px; align-items:center; }}
  .feat-row .rank {{ color:var(--muted); }}
  .feat-row .name {{ color:var(--text); }}
  .feat-row .meaning {{ color:var(--muted); font-size:11px; }}
  .feat-row .value {{ text-align:right; font-family:monospace; }}
  .feat-row .pctbar {{ background:var(--panel2); height:8px; border-radius:4px;
                       position:relative; overflow:hidden; }}
  .feat-row .pctbar > span {{ display:block; height:100%; background:var(--accent); }}
  .empty {{ color:var(--muted); text-align:center; padding:48px 0; font-size:13px; }}
</style>
</head>
<body>

<header>
  <h1>A股退市风险预测</h1>
  <p>基于历史 {d['n_delisted_total']} 例退市事件训练的 LightGBM 二分类，预测在册 {d['n_listed']:,} 只股票未来 365 日内退市概率</p>
  <p style="margin-top:4px"><b>预测快照日：</b> {d.get('inference_snapshot')}</p>
</header>

<nav>
  <a href="#overview">概览</a>
  <a href="#dist">风险分布</a>
  <a href="#importance">特征重要性</a>
  <a href="#table">全部股票排行</a>
  <a href="#year">逐年退市</a>
</nav>

<section id="overview">
  <h2>核心指标</h2>
  <p class="subtitle">基于 {test.get('n', 0):,} 个 2024+ 测试样本 ({test.get('n_pos', 0)} 正例) 的样本外评估</p>
  <div class="cards">{cards_html}</div>
</section>

<section id="dist">
  <h2>预测概率分布</h2>
  <p class="subtitle">在册全部 {len(predictions):,} 只股票的 P(下一年内退市) 直方图</p>
  <div id="dist_chart" style="height:340px"></div>
</section>

<section id="importance">
  <h2>特征重要性 (LightGBM gain)</h2>
  <p class="subtitle">用于预测的特征及其相对贡献。点击下方表格中的股票可在右侧看到该股每个特征的具体值。</p>
  <div id="imp_chart" style="height:520px"></div>
</section>

<section id="table">
  <h2>全部在册股票按风险排序</h2>
  <p class="subtitle">点击行可在右侧查看该股全部 {len(feature_importance)} 个特征的取值（按重要性排序）。</p>
  <div class="controls">
    <span>筛选：</span>
    <input type="text" id="search" placeholder="代码或名称…" style="min-width:200px;">
    <select id="tier_filter">
      <option value="all">全部风险等级</option>
      <option value="high">高风险 (≥30%)</option>
      <option value="med">中风险 (5-30%)</option>
      <option value="low">低风险 (&lt;5%)</option>
    </select>
    <span style="color:var(--muted)" id="row_count"></span>
  </div>
  <div class="grid-2">
    <div class="table-wrap" style="max-height:680px;">
      <table id="rank_table">
        <thead><tr>
          <th data-key="ts_code">代码</th>
          <th data-key="name">名称</th>
          <th data-key="p_delist">P(退市)</th>
          <th data-key="risk">等级</th>
          <th data-key="is_st">ST</th>
          <th data-key="days_listed">上市天数</th>
        </tr></thead>
        <tbody id="rank_body"></tbody>
      </table>
    </div>
    <div class="stockdetail" id="stock_detail">
      <div class="empty">点击左侧任一行查看该股全部特征</div>
    </div>
  </div>
</section>

<section id="year">
  <h2>历史逐年退市数</h2>
  <div class="grid-2">
    <div id="year_chart" style="height:340px"></div>
    <div class="table-wrap" style="max-height:340px;">
      <table>
        <thead><tr><th>年份</th><th>退市数</th></tr></thead>
        <tbody>{year_table_rows}</tbody>
      </table>
    </div>
  </div>
</section>

<script>
const DATA = {payload};
const FEAT_MEANING = {feat_meaning_json};
const LAYOUT = {{
  paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
  font: {{ color:'#c9d1d9', family:'PingFang SC, Microsoft YaHei, sans-serif', size:12 }},
  xaxis: {{ gridcolor:'#2a313c', zerolinecolor:'#2a313c' }},
  yaxis: {{ gridcolor:'#2a313c', zerolinecolor:'#2a313c' }},
  margin: {{ l:60, r:30, t:20, b:50 }},
}};
const CFG = {{ displaylogo:false, responsive:true }};
function merge(a,b){{ const o=JSON.parse(JSON.stringify(a)); for(const k in b){{ if(b[k]&&typeof b[k]==='object'&&!Array.isArray(b[k])) o[k]=merge(o[k]||{{}},b[k]); else o[k]=b[k]; }} return o; }}

// ---- Risk distribution ----
(function() {{
  const probs = DATA.predictions.map(r => r.p_delist);
  Plotly.newPlot('dist_chart', [{{
    x: probs, type:'histogram', nbinsx: 50,
    marker: {{ color: probs.map(p => p>=0.3?'#da3633':(p>=0.05?'#d29922':'#2ea043')) }},
  }}], merge(LAYOUT, {{
    xaxis: {{ title:'P(下一年内退市)', range:[0, 1] }},
    yaxis: {{ title:'股票数', type:'log' }},
  }}), CFG);
}})();

// ---- Feature importance ----
const IMP_SORTED = (DATA.feature_importance || []).slice();
IMP_SORTED.sort((a,b) => b.importance - a.importance);
const IMP_RANK = {{}};  // feature -> rank (0-indexed)
IMP_SORTED.forEach((r,i) => IMP_RANK[r.feature] = i);
(function() {{
  const top = IMP_SORTED.slice(0, 30).reverse();
  Plotly.newPlot('imp_chart', [{{
    x: top.map(r => r.importance),
    y: top.map(r => `${{r.feature}}  ·  ${{FEAT_MEANING[r.feature]||''}}`.slice(0, 60)),
    type:'bar', orientation:'h',
    marker: {{ color:'#58a6ff' }},
    hovertemplate: '%{{y}}<br>gain: %{{x}}<extra></extra>',
  }}], merge(LAYOUT, {{
    xaxis: {{ title:'gain' }},
    yaxis: {{ automargin:true }},
    margin: {{ l:340, r:30, t:20, b:50 }},
  }}), CFG);
}})();

// ---- Yearly delistings ----
(function() {{
  const ymap = DATA.delistings_by_year || {{}};
  const years = Object.keys(ymap).sort();
  const ys = years.filter(y => y >= '2010');
  Plotly.newPlot('year_chart', [{{
    x: ys, y: ys.map(y => ymap[y]),
    type:'bar', marker: {{ color:'#da3633' }},
    text: ys.map(y => ymap[y]), textposition:'outside',
  }}], merge(LAYOUT, {{
    yaxis: {{ title:'当年退市数' }},
    xaxis: {{ title:'年份', tickangle:-45 }},
  }}), CFG);
}})();

// ---- Sortable table + per-stock detail ----
(function() {{
  const tbody = document.getElementById('rank_body');
  const rowCount = document.getElementById('row_count');
  const search = document.getElementById('search');
  const tierFilter = document.getElementById('tier_filter');
  const detail = document.getElementById('stock_detail');

  let sortKey = 'p_delist';
  let sortDir = -1;
  let visibleRows = [];
  // Pre-sort by p_delist desc
  let allRows = DATA.predictions.slice();

  // Compute per-feature percentile (rank in population) for the selected stock detail view
  // Pre-sort each feature's values once
  const FEAT_SORTED = {{}};
  for (const f of DATA.feature_columns) {{
    const vals = allRows.map(r => r[f]).filter(v => v !== null && v !== undefined && !isNaN(v));
    vals.sort((a,b) => a-b);
    FEAT_SORTED[f] = vals;
  }}
  function pctRank(feat, v) {{
    if (v === null || v === undefined || isNaN(v)) return null;
    const arr = FEAT_SORTED[feat];
    if (!arr || !arr.length) return null;
    // binary search
    let lo=0, hi=arr.length;
    while (lo<hi) {{ const m=(lo+hi)>>>1; if (arr[m] < v) lo=m+1; else hi=m; }}
    return lo / arr.length;
  }}

  function tier(p) {{
    if (p >= 0.3) return 'high';
    if (p >= 0.05) return 'med';
    return 'low';
  }}

  function applyFilters() {{
    const q = search.value.trim().toLowerCase();
    const tf = tierFilter.value;
    visibleRows = allRows.filter(r => {{
      if (q && !(r.ts_code.toLowerCase().includes(q) || (r.name||'').toLowerCase().includes(q))) return false;
      if (tf !== 'all' && tier(r.p_delist) !== tf) return false;
      return true;
    }});
    visibleRows.sort((a,b) => {{
      let av = a[sortKey], bv = b[sortKey];
      if (sortKey === 'risk') {{ av = a.p_delist; bv = b.p_delist; }}
      if (typeof av === 'string') return av.localeCompare(bv) * sortDir;
      return ((av ?? -1) - (bv ?? -1)) * sortDir;
    }});
    renderTable();
  }}

  function renderTable() {{
    rowCount.textContent = `· 共 ${{visibleRows.length}} / ${{allRows.length}} 只`;
    // Limit to 500 rows for DOM perf, after sort
    const display = visibleRows.slice(0, 500);
    let h = '';
    for (const r of display) {{
      const t = tier(r.p_delist);
      const tlabel = {{high:'高', med:'中', low:'低'}}[t];
      const stFlag = r.is_st > 0 ? (r.is_starred_st > 0 ? '*ST' : 'ST') : '';
      h += `<tr data-code="${{r.ts_code}}">
        <td>${{r.ts_code}}</td>
        <td>${{r.name||''}}</td>
        <td><b>${{(r.p_delist*100).toFixed(2)}}%</b></td>
        <td><span class="pill ${{t}}">${{tlabel}}</span></td>
        <td>${{stFlag}}</td>
        <td>${{r.days_listed||''}}</td>
      </tr>`;
    }}
    tbody.innerHTML = h;
    if (visibleRows.length > 500) {{
      tbody.insertAdjacentHTML('beforeend',
        `<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:14px">
          (筛选/排序后显示前 500 行,继续筛选可缩小范围)</td></tr>`);
    }}
    // Re-attach row click handlers
    tbody.querySelectorAll('tr[data-code]').forEach(tr => {{
      tr.onclick = () => selectStock(tr.dataset.code, tr);
    }});
  }}

  function selectStock(code, tr) {{
    document.querySelectorAll('#rank_body tr.selected').forEach(t => t.classList.remove('selected'));
    if (tr) tr.classList.add('selected');
    const r = allRows.find(x => x.ts_code === code);
    if (!r) return;
    let h = `<h3>${{r.ts_code}} · ${{r.name||''}}</h3>
             <p class="head-line">P(退市) = <b style="color:var(--up)">${{(r.p_delist*100).toFixed(2)}}%</b>
             · 风险等级: <span class="pill ${{tier(r.p_delist)}}">${{ {{high:'高', med:'中', low:'低'}}[tier(r.p_delist)] }}</span></p>`;
    h += `<div style="font-size:11px;color:var(--muted);margin-bottom:6px">
            特征按全局重要性排序; 进度条 = 该值在 ${{allRows.length}} 只股票中的百分位</div>`;
    for (const item of IMP_SORTED) {{
      const f = item.feature;
      const v = r[f];
      let valStr = '—';
      if (v !== null && v !== undefined && !isNaN(v)) {{
        valStr = (Math.abs(v) >= 100 || Number.isInteger(v)) ? Number(v).toFixed(0) : Number(v).toFixed(3);
      }}
      const pr = pctRank(f, v);
      const pctStr = (pr !== null) ? `<div class="pctbar"><span style="width:${{(pr*100).toFixed(1)}}%"></span></div>` : '<div></div>';
      h += `<div class="feat-row">
        <div class="rank">${{IMP_RANK[f]+1}}</div>
        <div>
          <div class="name">${{f}}</div>
          <div class="meaning">${{FEAT_MEANING[f]||''}}</div>
        </div>
        <div class="value">${{valStr}}</div>
        ${{pctStr}}
      </div>`;
    }}
    detail.innerHTML = h;
  }}

  // Sort handlers
  document.querySelectorAll('#rank_table th').forEach(th => {{
    th.onclick = () => {{
      const k = th.dataset.key;
      if (sortKey === k) sortDir = -sortDir; else {{ sortKey = k; sortDir = (k === 'p_delist' || k === 'days_listed') ? -1 : 1; }}
      applyFilters();
    }};
  }});
  search.oninput = applyFilters;
  tierFilter.onchange = applyFilters;
  applyFilters();
  // Auto-select the highest-risk stock to populate the detail panel
  if (allRows.length) {{
    const top = allRows.slice().sort((a,b)=>b.p_delist - a.p_delist)[0];
    setTimeout(() => {{
      const tr = document.querySelector(`#rank_body tr[data-code="${{top.ts_code}}"]`);
      selectStock(top.ts_code, tr);
    }}, 0);
  }}
}})();
</script>
</body>
</html>"""
    return HTML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "stock_data" / "delist_predictions.json"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "delist.html"))
    args = ap.parse_args()
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    html = build_html(data)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html, encoding="utf-8")
    sz = out_p.stat().st_size
    print(f"Saved -> {out_p}  ({sz/1024:.1f} KB)")
    print(f"Open: file:///{out_p.resolve().as_posix()}")


if __name__ == "__main__":
    main()
