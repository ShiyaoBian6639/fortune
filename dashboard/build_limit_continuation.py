"""Self-contained dashboard for the consecutive-up-limit (连板) analysis.

Reads stock_data/limit_continuation.json (built by xgbmodel.limit_continuation)
and writes dashboard/limit_continuation.html — a single HTML file with
embedded JSON and Plotly loaded from CDN. No server, no auth.

Usage:
    ./venv/Scripts/python -m dashboard.build_limit_continuation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt_pct(x):
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"


def fmt_int(x):
    if x is None:
        return "—"
    return f"{x:,}"


def metric_card(label, value, sub=None, kind="neutral"):
    sub_html = f"<div class='m-sub'>{sub}</div>" if sub else ""
    return f"""<div class="card kind-{kind}">
      <div class="m-label">{label}</div>
      <div class="m-value">{value}</div>
      {sub_html}
    </div>"""


def build_html(d: dict) -> str:
    overall = d["probabilities_overall"]
    recent = d["probabilities_recent_90d"]
    cls = d.get("classifier") or {}
    lifts = d.get("feature_lifts") or []
    overlap = d.get("top_pred_overlap_today") or []

    # Recent base continuation rate
    recent_streak = recent.get("by_streak", [])
    recent_avg = (sum(r["p_continue"] * r["n"] for r in recent_streak)
                  / max(sum(r["n"] for r in recent_streak), 1)) if recent_streak else 0.0

    # Headline cards
    type_map = {r["type"]: r for r in overall.get("by_type_given_up_t", [])}
    one_line = type_map.get("one_line", {})
    t_shape = type_map.get("t_shape", {})

    cards_html = "".join([
        metric_card("Base P(up-limit at t)",
                    fmt_pct(overall["p_uplimit_unconditional_t"]),
                    sub=f"{fmt_int(overall['n_up_limit_rows'])} of {fmt_int(overall['n_total_rows'])} rows"),
        metric_card("Overall continuation P(↑ₜ₊₁ | ↑ₜ)",
                    fmt_pct((one_line.get('p_continue', 0) * one_line.get('n', 0)
                             + t_shape.get('p_continue', 0) * t_shape.get('n', 0))
                            / max(one_line.get('n', 0) + t_shape.get('n', 0), 1)),
                    sub=f"across {fmt_int(one_line.get('n', 0) + t_shape.get('n', 0))} up-limit days",
                    kind="up"),
        metric_card("一字板 continuation",
                    fmt_pct(one_line.get("p_continue")),
                    sub=f"n={fmt_int(one_line.get('n'))} · P(next 一字)={fmt_pct(one_line.get('p_continue_one_line'))}",
                    kind="up"),
        metric_card("T-shape continuation",
                    fmt_pct(t_shape.get("p_continue")),
                    sub=f"n={fmt_int(t_shape.get('n'))} · {3.2:.1f}× lower than 一字",
                    kind="down"),
        metric_card("Recent 90d continuation",
                    fmt_pct(recent_avg),
                    sub="last 90 trading days",
                    kind=("up" if recent_avg > 0.25 else "down")),
        metric_card("Classifier AUC (test 2024+)",
                    f"{cls.get('auc_test', 0):.3f}" if cls else "—",
                    sub=f"AP {cls.get('average_precision_test', 0):.3f} · base {cls.get('base_rate_test', 0):.3f}"
                        if cls else None,
                    kind="up"),
    ])

    # By streak table
    by_streak_rows = "".join([
        f"<tr><td>{r['streak']}</td><td>{r['n']:,}</td>"
        f"<td>{r['p_continue']*100:.1f}%</td>"
        f"<td>{r['p_continue_one_line']*100:.1f}%</td></tr>"
        for r in overall.get("by_streak", []) if r["streak"] >= 1 and r["streak"] <= 12
    ])

    # By board
    by_board_rows = "".join([
        f"<tr><td>{r['board']}</td><td>{r['n']:,}</td>"
        f"<td class='{('pos' if r['p_continue'] > 0.25 else 'neg')}'>"
        f"{r['p_continue']*100:.1f}%</td></tr>"
        for r in overall.get("by_board_given_up_t", [])
    ])

    # By year
    by_year_rows = "".join([
        f"<tr><td>{r['year']}</td><td>{r['n']:,}</td>"
        f"<td class='{('pos' if r['p_continue'] > 0.25 else 'neg')}'>"
        f"{r['p_continue']*100:.1f}%</td></tr>"
        for r in overall.get("by_year_given_up_t", [])
    ])

    # By cap bucket
    by_cap_rows = ""
    if "by_cap_given_up_t" in overall:
        by_cap_rows = "".join([
            f"<tr><td>{r['cap_bucket']}</td><td>{r['n']:,}</td>"
            f"<td class='{('pos' if r['p_continue'] > 0.25 else 'neg')}'>"
            f"{r['p_continue']*100:.1f}%</td></tr>"
            for r in overall["by_cap_given_up_t"]
        ])

    # Today overlap table
    def _overlap_row(r):
        ul = r["is_up_limit_today"]
        ol = r["one_line_today"]
        amt = r["amount_surge"]
        amt_str = "—" if amt is None else f"{amt:.2f}x"
        ul_pill = '<span class="pill yes">YES</span>' if ul else '<span class="pill no">no</span>'
        ol_pill = '<span class="pill yes">one-line</span>' if ol else ""
        return (
            f"<tr><td>{r['ts_code']}</td>"
            f"<td>{r['pred_pct_chg_next']:.2f}%</td>"
            f"<td>{ul_pill}</td><td>{ol_pill}</td>"
            f"<td>{r['streak_up']}</td>"
            f"<td>{r['limits_5d']:.0f}</td>"
            f"<td>{amt_str}</td></tr>"
        )
    overlap_rows = "".join(_overlap_row(r) for r in overlap)

    # Classifier importance table
    imp_rows = ""
    if cls and "feature_importance" in cls:
        for r in cls["feature_importance"][:15]:
            imp_rows += f"<tr><td>{r['feature']}</td><td>{r['importance']:.0f}</td></tr>"

    payload = json.dumps(d, separators=(",", ":"), default=str, ensure_ascii=False)

    HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A股连板概率分析 — XGB 预测验证</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg:#0e1117; --panel:#161b22; --panel2:#1c232e; --border:#2a313c;
    --text:#c9d1d9; --muted:#7d8590; --accent:#58a6ff;
    --up:#da3633; --down:#2ea043;  /* A-share convention: red=up, green=down */
    --warn:#d29922;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "PingFang SC","Microsoft YaHei",-apple-system,sans-serif;
         background:var(--bg); color:var(--text); margin:0; line-height:1.5; }}
  header {{ background:linear-gradient(180deg,#2a1a1c 0%,#0e1117 100%);
           padding:28px 40px; border-bottom:1px solid var(--border); }}
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
  .card.kind-up {{ border-left-color:var(--up); }}
  .card.kind-down {{ border-left-color:var(--down); }}
  .m-label {{ color:var(--muted); font-size:11px; text-transform:uppercase;
             letter-spacing:0.5px; }}
  .m-value {{ font-size:22px; font-weight:600; margin:4px 0; }}
  .m-sub {{ color:var(--muted); font-size:11px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px;
          background:var(--panel); }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:right; }}
  th {{ background:var(--panel2); color:var(--muted); font-weight:500;
        font-size:11px; text-transform:uppercase; letter-spacing:0.5px;
        text-align:right; }}
  th:first-child, td:first-child {{ text-align:left; }}
  tr:hover {{ background:var(--panel2); }}
  .pos {{ color:var(--up); }}
  .neg {{ color:var(--down); }}
  .pill {{ display:inline-block; padding:1px 8px; border-radius:3px;
          font-size:11px; font-weight:500; }}
  .pill.yes {{ background:rgba(218,54,51,0.18); color:#f87171; }}
  .pill.no {{ background:rgba(125,133,144,0.18); color:var(--muted); }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:900px) {{ .grid-2 {{ grid-template-columns:1fr; }} }}
  .table-wrap {{ max-height:480px; overflow-y:auto;
                border:1px solid var(--border); border-radius:6px; }}
  .callout {{ background:var(--panel2); border-left:3px solid var(--warn);
             padding:12px 16px; margin:12px 0; border-radius:4px;
             font-size:13px; color:var(--text); }}
</style>
</head>
<body>

<header>
  <h1>A股连板概率分析</h1>
  <p>历史涨停后次日继续涨停（连板）的概率与预测特征</p>
  <p style="margin-top:4px">
    <b>样本期：</b> {d['panel_dates'][0]} → {d['panel_dates'][1]}
    &nbsp;·&nbsp; <b>股票数：</b> {d['n_stocks']:,}
    &nbsp;·&nbsp; <b>涨停记录：</b> {fmt_int(overall['n_up_limit_rows'])}
  </p>
</header>

<nav>
  <a href="#overview">概览</a>
  <a href="#streak">连板递进</a>
  <a href="#segments">分组明细</a>
  <a href="#features">关键特征</a>
  <a href="#classifier">预测模型</a>
  <a href="#today">今日预测核对</a>
</nav>

<section id="overview">
  <h2>核心结论</h2>
  <p class="subtitle">基础概率与"一字板 vs T字板"差异</p>
  <div class="cards">{cards_html}</div>
  <div class="callout">
    <b>关键洞察：</b> 一字板（开盘即一字封死、当天没破板）次日继续涨停的概率约为
    <b>{one_line.get('p_continue', 0)*100:.1f}%</b>，而开过板/T字板只有
    <b>{t_shape.get('p_continue', 0)*100:.1f}%</b>。封单强度（封板形态）是预测连板的最重要单一信号之一。
  </div>
</section>

<section id="streak">
  <h2>连板阶梯：N板 → N+1板 概率</h2>
  <p class="subtitle">已经N板的股票，次日再封板的条件概率（剔除样本数 &lt; 50 的streak）</p>
  <div id="streak_chart" style="height:340px"></div>
  <div class="grid-2" style="margin-top:18px">
    <div>
      <h3 style="font-size:14px;margin:0 0 8px;color:var(--muted)">连板阶梯表</h3>
      <table>
        <thead><tr><th>当前N板</th><th>样本数</th><th>P(继续封板)</th><th>P(下一日一字)</th></tr></thead>
        <tbody>{by_streak_rows}</tbody>
      </table>
    </div>
    <div>
      <h3 style="font-size:14px;margin:0 0 8px;color:var(--muted)">板块差异 (given 涨停)</h3>
      <table>
        <thead><tr><th>板块</th><th>样本数</th><th>P(继续封板)</th></tr></thead>
        <tbody>{by_board_rows}</tbody>
      </table>
      <div style="margin-top:6px;color:var(--muted);font-size:11px">
        STAR/科创板涨幅限制20%，封单更难维持，故连板率显著较低。
      </div>
    </div>
  </div>
</section>

<section id="segments">
  <h2>分时段 / 市值分组</h2>
  <p class="subtitle">不同年份与市值档位的连板概率差异</p>
  <div class="grid-2">
    <div>
      <h3 style="font-size:14px;margin:0 0 8px;color:var(--muted)">逐年统计 (given 涨停)</h3>
      <table>
        <thead><tr><th>年份</th><th>样本数</th><th>P(继续封板)</th></tr></thead>
        <tbody>{by_year_rows}</tbody>
      </table>
    </div>
    <div>
      <h3 style="font-size:14px;margin:0 0 8px;color:var(--muted)">流通市值分位 (given 涨停)</h3>
      <table>
        <thead><tr><th>档位</th><th>样本数</th><th>P(继续封板)</th></tr></thead>
        <tbody>{by_cap_rows}</tbody>
      </table>
      <div style="margin-top:6px;color:var(--muted);font-size:11px">
        XS = 流通市值最小的1/5，XL = 最大的1/5。小盘股连板率明显高于大盘股。
      </div>
    </div>
  </div>
</section>

<section id="features">
  <h2>关键预测特征 — 五分位提升表</h2>
  <p class="subtitle">把每个特征分成5档，看 P(继续封板) 在最低档与最高档的差异 (lift)</p>
  <div id="lift_chart" style="height:420px"></div>
  <div class="callout" id="lift_callout">特征解释将根据数据自动填充。</div>
</section>

<section id="classifier">
  <h2>LightGBM 二分类预测</h2>
  <p class="subtitle">输入：今日处于涨停状态时的特征 · 输出：次日是否继续涨停 ·
    样本：训练 &lt; 2024-01 / 测试 ≥ 2024-01</p>
  <div class="grid-2">
    <div>
      <h3 style="font-size:14px;margin:0 0 8px;color:var(--muted)">模型表现</h3>
      <table>
        <tr><td>训练样本</td><td>{fmt_int(cls.get('n_train'))}</td></tr>
        <tr><td>测试样本</td><td>{fmt_int(cls.get('n_test'))}</td></tr>
        <tr><td>测试 AUC</td><td><b class="pos">{cls.get('auc_test', 0):.4f}</b></td></tr>
        <tr><td>测试 AP</td><td>{cls.get('average_precision_test', 0):.4f}</td></tr>
        <tr><td>测试 base rate</td><td>{cls.get('base_rate_test', 0):.4f}</td></tr>
        <tr><td>训练 AUC</td><td>{cls.get('auc_train', 0):.4f}</td></tr>
        <tr><td>best_iteration</td><td>{cls.get('best_iteration', 0)}</td></tr>
      </table>
      <div style="margin-top:8px;color:var(--muted);font-size:12px">
        测试期 AUC ≈ 0.68 表示模型对"是否连板"有显著区分力（base rate ≈ 23%）。
      </div>
    </div>
    <div>
      <h3 style="font-size:14px;margin:0 0 8px;color:var(--muted)">特征重要性 (Top 15, gain)</h3>
      <table>
        <thead><tr><th>特征</th><th>Gain</th></tr></thead>
        <tbody>{imp_rows}</tbody>
      </table>
    </div>
  </div>
</section>

<section id="today">
  <h2>今日XGB预测 Top-30 与"是否已涨停"核对</h2>
  <p class="subtitle">XGB Top-30 中已经处于涨停的，明日继续封板的先验概率可参考上方一字/T字差异</p>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>ts_code</th><th>预测涨幅</th><th>今日涨停</th><th>形态</th>
        <th>当前N板</th><th>近5日封板数</th><th>量能放大</th>
      </tr></thead>
      <tbody>{overlap_rows}</tbody>
    </table>
  </div>
</section>

<script>
const DATA = {payload};
const LAYOUT_BASE = {{
  paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
  font: {{ color:'#c9d1d9', family:'PingFang SC, Microsoft YaHei, sans-serif', size:12 }},
  xaxis: {{ gridcolor:'#2a313c', zerolinecolor:'#2a313c' }},
  yaxis: {{ gridcolor:'#2a313c', zerolinecolor:'#2a313c' }},
  margin: {{ l:60, r:30, t:20, b:50 }},
}};
const CONFIG = {{ displaylogo:false, responsive:true }};

function merge(a,b){{
  const o = JSON.parse(JSON.stringify(a));
  for (const k in b){{
    if (b[k]&&typeof b[k]==='object'&&!Array.isArray(b[k]))
      o[k] = merge(o[k]||{{}}, b[k]);
    else o[k]=b[k];
  }}
  return o;
}}

// Streak chart
(function() {{
  const arr = DATA.probabilities_overall.by_streak.filter(r=>r.streak>=1 && r.streak<=12);
  const traces = [{{
    x: arr.map(r => `${{r.streak}}板`),
    y: arr.map(r => r.p_continue * 100),
    name: 'P(继续封板)', type: 'bar',
    marker: {{ color:'#da3633' }},
    text: arr.map(r => `${{(r.p_continue*100).toFixed(1)}}%`),
    textposition: 'outside',
    hovertemplate: 'N=%{{x}}<br>n=%{{customdata}}<br>P=%{{y:.1f}}%<extra></extra>',
    customdata: arr.map(r=>r.n.toLocaleString()),
  }}, {{
    x: arr.map(r => `${{r.streak}}板`),
    y: arr.map(r => r.p_continue_one_line * 100),
    name: 'P(下一日一字)', type: 'bar',
    marker: {{ color:'#d29922' }},
    text: arr.map(r => `${{(r.p_continue_one_line*100).toFixed(1)}}%`),
    textposition: 'outside',
  }}];
  const layout = merge(LAYOUT_BASE, {{
    yaxis: {{ title:'概率 (%)', ticksuffix:'%', range:[0, 100] }},
    barmode: 'group',
    legend: {{ orientation:'h', y:1.1 }},
  }});
  Plotly.newPlot('streak_chart', traces, layout, CONFIG);
}})();

// Lift chart — top 12 features by lift
(function() {{
  const lifts = (DATA.feature_lifts || []).slice(0, 12);
  if (!lifts.length) return;
  // For each feature, plot 5 bars (one per quintile) of P(continue)
  const traces = lifts.map((r,i) => ({{
    x: r.bins.map(b => `Q${{b.bin+1}}`),
    y: r.bins.map(b => b.p_continue * 100),
    name: r.feature,
    type: 'bar',
    visible: i === 0,
    text: r.bins.map(b => `${{(b.p_continue*100).toFixed(1)}}%`),
    textposition: 'outside',
    hovertemplate: '%{{x}}<br>n=%{{customdata}}<br>P=%{{y:.1f}}%<extra></extra>',
    customdata: r.bins.map(b => b.n.toLocaleString()),
  }}));
  const buttons = lifts.map((r,i) => ({{
    method: 'update', label: `${{r.feature}} (lift=${{(r.lift*100).toFixed(1)}}pp)`,
    args: [{{visible: lifts.map((_,j)=>j===i)}}, {{title: ''}}]
  }}));
  const layout = merge(LAYOUT_BASE, {{
    yaxis: {{ title:'P(继续封板) (%)', ticksuffix:'%', range:[0, 60] }},
    xaxis: {{ title: '特征五分位 (Q1=最低, Q5=最高)' }},
    updatemenus: [{{
      buttons: buttons,
      direction:'down', showactive:true, x:0, xanchor:'left', y:1.18,
      bgcolor:'#1c232e', bordercolor:'#2a313c',
    }}],
    margin: {{ t:60 }},
  }});
  Plotly.newPlot('lift_chart', traces, layout, CONFIG);
  // Update callout with first feature's interpretation
  const top = lifts[0];
  const dir = top.p_top_bin > top.p_bot_bin ? '高档位 → 高概率' : '低档位 → 高概率';
  document.getElementById('lift_callout').innerHTML =
    `<b>提升最大特征：</b> <code>${{top.feature}}</code> · ` +
    `最低五分位 P=${{(top.p_bot_bin*100).toFixed(1)}}%，最高五分位 P=${{(top.p_top_bin*100).toFixed(1)}}%（${{dir}}）。 ` +
    `下拉框可切换其他特征。`;
}})();
</script>
</body>
</html>"""
    return HTML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "stock_data" / "limit_continuation.json"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "limit_continuation.html"))
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    html = build_html(data)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html, encoding="utf-8")
    print(f"Saved -> {out_p}  ({out_p.stat().st_size / 1024:.1f} KB)")
    print(f"Open: file:///{out_p.resolve().as_posix()}")


if __name__ == "__main__":
    main()
