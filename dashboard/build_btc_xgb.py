"""Build a self-contained BTC XGBoost backtest dashboard.

Reads btc_data/backtest_results.json and writes dashboard/btc_xgb.html — a
single HTML file with embedded JSON and Plotly loaded from CDN. No server,
no password, no auth: just open the file in a browser.

Usage:
    ./venv/Scripts/python -m dashboard.build_btc_xgb
"""

import argparse
import html
import json
from pathlib import Path

EXIT_COLORS = {
    "take_profit": "#16a34a",
    "stop_loss":   "#dc2626",
    "time_stop":   "#d97706",
    "signal_flip": "#7c3aed",
}


def fmt_pct(x, signed=False):
    if x is None:
        return "—"
    s = f"{x * 100:+.2f}%" if signed else f"{x * 100:.2f}%"
    return s


def fmt_num(x, dp=2):
    if x is None:
        return "—"
    return f"{x:.{dp}f}"


def metric_card(label, value, sub=None, kind="neutral"):
    sub_html = f"<div class='m-sub'>{sub}</div>" if sub else ""
    return f"""<div class="card kind-{kind}">
      <div class="m-label">{label}</div>
      <div class="m-value">{value}</div>
      {sub_html}
    </div>"""


def build_html(data: dict) -> str:
    m = data["metrics"]
    cfg = data["config"]
    rng = data["date_range"]

    pf_str = "∞" if m["profit_factor"] is None else fmt_num(m["profit_factor"])
    ic_raw = data.get("ic_raw", data.get("ic"))
    ic_norm = data.get("ic_norm")
    rmse = data.get("rmse_raw", data.get("rmse"))
    best_params = data.get("best_params", {})
    tuning_results = data.get("tuning_results", [])
    imp_stability = data.get("importance_stability", [])
    regime_enabled = data.get("regime_filter_enabled", False)
    regime_blocked = data.get("regime_blocked_bars", 0)

    cards_html = "".join([
        metric_card("Total Return",
                    fmt_pct(m["total_return"], signed=True),
                    sub=f"B&amp;H: {fmt_pct(m['buy_hold_return'], signed=True)}",
                    kind="up" if m["total_return"] >= 0 else "down"),
        metric_card("CAGR",
                    fmt_pct(m["cagr"], signed=True),
                    sub=f"B&amp;H: {fmt_pct(m['buy_hold_cagr'], signed=True)}",
                    kind="up" if m["cagr"] >= 0 else "down"),
        metric_card("Sharpe",
                    fmt_num(m["sharpe"]),
                    sub=f"B&amp;H: {fmt_num(m['buy_hold_sharpe'])}",
                    kind="up" if m["sharpe"] >= 0 else "down"),
        metric_card("Sortino",
                    fmt_num(m["sortino"]),
                    sub=f"down-only vol",
                    kind="up" if m["sortino"] >= 0 else "down"),
        metric_card("Max Drawdown",
                    fmt_pct(m["max_drawdown"], signed=True),
                    sub=f"B&amp;H: {fmt_pct(m['buy_hold_max_dd'], signed=True)}",
                    kind="down"),
        metric_card("Win Rate",
                    fmt_pct(m["win_rate"]),
                    sub=f"avg win {fmt_pct(m['avg_win'], signed=True)} / loss {fmt_pct(m['avg_loss'], signed=True)}",
                    kind="up" if m["win_rate"] >= 0.5 else "down"),
        metric_card("Profit Factor",
                    pf_str,
                    sub=f"sum wins / |sum losses|",
                    kind="up" if (m["profit_factor"] or 0) >= 1.0 else "down"),
        metric_card("Trades",
                    f"{m['n_trades']}",
                    sub=f"avg hold {fmt_num(m['avg_holding_days'], 1)}d",
                    kind="neutral"),
        metric_card("IC (raw)",
                    fmt_num(ic_raw, 4),
                    sub=f"IC norm {fmt_num(ic_norm, 4)} · RMSE {fmt_num(rmse, 4)}",
                    kind="up" if (ic_raw or 0) > 0 else "down"),
        metric_card("Regime Filter",
                    f"{regime_blocked} blocked" if regime_enabled else "off",
                    sub=f"{regime_blocked / max(len(data['equity_curve']), 1) * 100:.1f}% of bars" if regime_enabled else "all bars tradable",
                    kind="neutral"),
    ])

    exit_breakdown = m["exit_breakdown"] or {}
    exits_html = "  ".join(
        f'<span class="dot" style="background:{EXIT_COLORS.get(k, "#888")}"></span>'
        f'<span>{k.replace("_", " ").title()}: <b>{v}</b></span>'
        for k, v in exit_breakdown.items()
    )

    per_model_ic = data.get("per_model_ic", {})
    sign_history = data.get("sign_history", [])
    ensemble_meta = data.get("ensemble_meta", {})
    val_ic_history = data.get("val_ic_history", [])
    method_comparison = data.get("method_comparison", [])
    best_method = data.get("best_method", "")
    model_comparison = data.get("model_comparison", [])
    best_model = data.get("best_model", "")
    feature_importance = data.get("feature_importance", []) or []
    has_sign_history = bool(data.get("sign_history"))
    has_importance = bool(imp_stability) or bool(feature_importance)

    if model_comparison:
        def _mrow(m):
            is_best = m["model"] == best_model
            cls = ' class="best"' if is_best else ""
            pf = m.get("profit_factor")
            pf_str = "∞" if pf is None else f"{pf:.2f}"
            star = " ★" if is_best else ""
            ic_norm = m.get("ic_norm_sign_corrected", 0)
            ic_raw = m.get("ic_raw_sign_corrected", 0)
            hr = m.get("hit_rate", float("nan"))
            return (
                f"<tr{cls}>"
                f"<td>{m['model']}{star}</td>"
                f"<td class=\"{('pos' if ic_norm > 0 else 'neg')}\">{ic_norm:+.4f}</td>"
                f"<td class=\"{('pos' if ic_raw > 0 else 'neg')}\">{ic_raw:+.4f}</td>"
                f"<td class=\"{('pos' if hr >= 0.5 else 'neg')}\">{hr * 100:.1f}%</td>"
                f"<td class=\"{('pos' if m['sharpe'] > 0 else 'neg')}\">{m['sharpe']:+.3f}</td>"
                f"<td class=\"{('pos' if m['total_return'] > 0 else 'neg')}\">{m['total_return']*100:+.2f}%</td>"
                f"<td>{pf_str}</td>"
                f"<td class=\"neg\">{m['max_drawdown']*100:+.2f}%</td>"
                f"<td>{m['n_trades']}</td>"
                f"<td>{m.get('elapsed_s', 0):.1f}s</td>"
                f"</tr>"
            )
        mrows = "".join(_mrow(m) for m in model_comparison)
        model_html = f"""<section id="models">
  <h2>Model Architecture Comparison</h2>
  <p class="subtitle">Identical input/output across architectures · ★ = best by Sharpe (≥15 trades) · IC and HR are AFTER trailing-OOS-IC sign correction</p>
  <div class="table-wrap" style="max-height:340px">
    <table class="trades">
      <thead><tr>
        <th>Model</th><th>IC (norm)</th><th>IC (raw)</th><th>Hit Rate</th>
        <th>Sharpe</th><th>Total</th><th>PF</th><th>Max DD</th><th>N</th><th>Time</th>
      </tr></thead>
      <tbody>{mrows}</tbody>
    </table>
  </div>
  <p style="font-size:12px;color:var(--muted);margin-top:8px">
    <b>xgb / lgb</b>: tabular gradient boosting, single-row input ·
    <b>mlp</b>: 2-layer GELU MLP, single-row input ·
    <b>dcnn</b>: dilated 1D causal CNN over 20-day window ·
    <b>transformer</b>: 2-layer self-attention encoder over 20-day window
  </p>
</section>"""
    else:
        model_html = ""

    if method_comparison:
        def _row(m):
            is_best = m["method"] == best_method
            cls = ' class="best"' if is_best else ""
            pf = m.get("profit_factor")
            pf_str = "∞" if pf is None else f"{pf:.2f}"
            star = " ★" if is_best else ""
            return (
                f"<tr{cls}>"
                f"<td>{m['method']}{star}</td>"
                f"<td class=\"{('pos' if m['ic_raw'] > 0 else 'neg')}\">{m['ic_raw']:+.4f}</td>"
                f"<td class=\"{('pos' if m['ic_norm'] > 0 else 'neg')}\">{m['ic_norm']:+.4f}</td>"
                f"<td class=\"{('pos' if m['sharpe'] > 0 else 'neg')}\">{m['sharpe']:+.3f}</td>"
                f"<td class=\"{('pos' if m['total_return'] > 0 else 'neg')}\">{m['total_return']*100:+.2f}%</td>"
                f"<td>{pf_str}</td>"
                f"<td class=\"neg\">{m['max_drawdown']*100:+.2f}%</td>"
                f"<td>{m['n_trades']}</td>"
                f"</tr>"
            )
        comp_rows = "".join(_row(m) for m in method_comparison)
        method_html = f"""<section id="methods">
  <h2>Ensemble Method Comparison</h2>
  <p class="subtitle">Six blending strategies tried; ★ = selected by Sharpe (≥15 trades)</p>
  <div class="table-wrap" style="max-height:340px">
    <table class="trades">
      <thead><tr>
        <th>Method</th><th>IC (raw)</th><th>IC (norm)</th>
        <th>Sharpe</th><th>Total</th><th>PF</th><th>Max DD</th><th>N</th>
      </tr></thead>
      <tbody>{comp_rows}</tbody>
    </table>
  </div>
  <p style="font-size:12px;color:var(--muted);margin-top:8px">
    <b>mean</b> = equal-weight average ·
    <b>ic_weighted</b> = weight by trailing |IC| ·
    <b>ic_filter</b> = drop models with |IC| &lt; 0.05 ·
    <b>top_k</b> = at each bar keep only top-k models by trailing |IC|
  </p>
</section>"""
    else:
        method_html = ""

    if per_model_ic:
        rows = "".join(
            f'<tr><td>{k}</td><td class="{("pos" if v > 0 else "neg")}">{fmt_num(v, 4)}</td></tr>'
            for k, v in per_model_ic.items()
        )
        per_model_html = f"""<section id="ensemble">
  <h2>Per-Model IC</h2>
  <p class="subtitle">Pearson IC of each base model vs primary normalized target (h={data.get('horizon', 5)})</p>
  <p style="font-size:13px;color:var(--muted)">
    Models: <span style="color:var(--text)">{', '.join(ensemble_meta.get('regression_models', []))}</span>
    · sample weights: exp decay half-life={ensemble_meta.get('weight_half_life', '?')}d
    · adaptive sign: <b style="color:var(--text)">{ensemble_meta.get('adaptive_sign', False)}</b>
  </p>
  <div class="table-wrap" style="max-height:280px">
    <table class="trades">
      <thead><tr><th>Model</th><th>IC</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>"""
    else:
        per_model_html = ""

    if tuning_results:
        sorted_tr = sorted(tuning_results,
                           key=lambda r: (r["mean_ic"] if not (r["mean_ic"] is None or
                                          (isinstance(r["mean_ic"], float) and r["mean_ic"] != r["mean_ic"])) else -1),
                           reverse=True)
        rows_html = "".join([
            f"<tr{' class=\"best\"' if i == 0 else ''}>"
            f"<td>{r['max_depth']}</td><td>{r['learning_rate']}</td>"
            f"<td>{r['n_estimators']}</td>"
            f"<td>{fmt_num(r['mean_ic'], 4)}</td>"
            f"<td>{', '.join(fmt_num(x, 3) for x in r.get('fold_ics', []))}</td></tr>"
            for i, r in enumerate(sorted_tr)
        ])
        bp_str = ", ".join(f"{k}={v}" for k, v in best_params.items()
                           if k in ("max_depth", "learning_rate", "n_estimators",
                                    "subsample", "colsample_bytree", "reg_alpha",
                                    "reg_lambda", "min_child_weight"))
        tuning_html = f"""<section id="tuning">
  <h2>Hyperparameter Tuning</h2>
  <p class="subtitle">TimeSeriesSplit (4 folds) on first 50% of data · {len(tuning_results)} grid points · normalized target</p>
  <p style="font-size:13px;color:var(--muted)">Best: <span style="color:var(--text)">{bp_str}</span></p>
  <div class="table-wrap" style="max-height:340px">
    <table class="trades">
      <thead><tr>
        <th>max_depth</th><th>lr</th><th>n_est</th><th>mean IC</th><th>fold ICs</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</section>"""
    else:
        tuning_html = ""

    payload = json.dumps(data, separators=(",", ":"), default=str)

    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BTC XGBoost Backtest Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg:        #0e1117;
    --panel:     #161b22;
    --panel2:    #1c232e;
    --border:    #2a313c;
    --text:      #c9d1d9;
    --muted:     #7d8590;
    --accent:    #58a6ff;
    --up:        #16a34a;
    --down:      #dc2626;
    --neutral:   #58a6ff;
    --warn:      #d97706;
    --purple:    #7c3aed;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg); color: var(--text);
    margin: 0; padding: 0; line-height: 1.5;
  }}
  header {{
    background: linear-gradient(180deg, #1a2330 0%, #0e1117 100%);
    padding: 28px 40px 20px; border-bottom: 1px solid var(--border);
  }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; font-weight: 600; }}
  header p  {{ margin: 0; color: var(--muted); font-size: 13px; }}
  nav {{
    position: sticky; top: 0; z-index: 10;
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 8px 40px; overflow-x: auto; white-space: nowrap;
  }}
  nav a {{
    display: inline-block; padding: 6px 14px; margin-right: 4px;
    color: var(--muted); text-decoration: none; font-size: 13px;
    border-radius: 4px;
  }}
  nav a:hover {{ background: var(--panel2); color: var(--text); }}
  section {{
    padding: 24px 40px; border-bottom: 1px solid var(--border);
  }}
  section h2 {{ font-size: 18px; margin: 0 0 4px; font-weight: 600; }}
  section .subtitle {{ color: var(--muted); font-size: 12px; margin: 0 0 16px; }}
  .cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 10px;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 14px;
  }}
  .card.kind-up {{ border-left: 3px solid var(--up); }}
  .card.kind-down {{ border-left: 3px solid var(--down); }}
  .card.kind-neutral {{ border-left: 3px solid var(--neutral); }}
  .m-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
             letter-spacing: 0.5px; }}
  .m-value {{ font-size: 22px; font-weight: 600; margin: 4px 0; }}
  .m-sub {{ color: var(--muted); font-size: 11px; }}
  .grid-2 {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }}
  @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .legend-row {{
    display: flex; flex-wrap: wrap; gap: 16px;
    align-items: center; margin-top: 12px;
    font-size: 12px; color: var(--muted);
  }}
  .dot {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 4px; vertical-align: middle;
  }}
  table.trades {{
    width: 100%; border-collapse: collapse; font-size: 12px;
    background: var(--panel);
  }}
  table.trades th, table.trades td {{
    padding: 8px 10px; border-bottom: 1px solid var(--border);
    text-align: right;
  }}
  table.trades th {{
    background: var(--panel2); color: var(--muted);
    font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.5px; font-size: 11px; cursor: pointer;
    user-select: none;
  }}
  table.trades th:first-child, table.trades td:first-child {{ text-align: left; }}
  table.trades tbody tr:hover {{ background: var(--panel2); }}
  table.trades tr.best td {{ background: rgba(22,163,74,0.10); font-weight: 600; }}
  .pos {{ color: var(--up); }}
  .neg {{ color: var(--down); }}
  .pill {{
    display: inline-block; padding: 1px 8px; border-radius: 3px;
    font-size: 11px; font-weight: 500;
  }}
  .pill.take_profit {{ background: rgba(22,163,74,.18); color: #4ade80; }}
  .pill.stop_loss   {{ background: rgba(220,38,38,.18); color: #f87171; }}
  .pill.time_stop   {{ background: rgba(217,119,6,.18); color: #fbbf24; }}
  .pill.signal_flip {{ background: rgba(124,58,237,.18); color: #a78bfa; }}
  .table-wrap {{
    max-height: 500px; overflow-y: auto;
    border: 1px solid var(--border); border-radius: 6px;
  }}
</style>
</head>
<body>

<header>
  <h1>BTC Multi-Model Ensemble Strategy Backtest</h1>
  <p>Walk-forward ensemble (XGB ×4 horizons + Ridge-QT + XGB-cls) with adaptive sign correction via trailing OOS IC. Vol-normalized targets, intraday microstructure features, regime gating.</p>
  <p style="margin-top:4px">
    <b>Period:</b> {rng['start']} → {rng['end']}
    &nbsp;·&nbsp; <b>Features:</b> {data['n_features']}
    &nbsp;·&nbsp; <b>Refits:</b> {data['refits']}
    &nbsp;·&nbsp; <b>Horizon:</b> {data['horizon']} days
    &nbsp;·&nbsp; <b>Ensemble IC:</b> {fmt_num(ic_raw, 4)} raw / {fmt_num(ic_norm, 4)} norm
    {f'&nbsp;·&nbsp; <b>Method:</b> <span style="color:#16a34a">{best_method}</span>' if best_method else ''}
    {f'&nbsp;·&nbsp; <b>Best model:</b> <span style="color:#16a34a">{best_model}</span>' if best_model else ''}
  </p>
</header>

<nav>
  <a href="#overview">Overview</a>
  <a href="#equity">Equity</a>
  <a href="#drawdown">Drawdown</a>
  <a href="#signals">Signals</a>
  <a href="#predictions">Predictions</a>
  {'<a href="#models">Models</a>' if model_comparison else ''}
  {'<a href="#methods">Methods</a>' if method_comparison else ''}
  {'<a href="#ensemble">Ensemble</a>' if per_model_ic else ''}
  {'<a href="#sign">Sign Correction</a>' if has_sign_history else ''}
  {'<a href="#importance">Importance</a>' if has_importance else ''}
  <a href="#trades">Trades</a>
</nav>

<section id="overview">
  <h2>Performance Summary</h2>
  <p class="subtitle">All metrics out-of-sample · capital ${cfg['initial_capital']:,.0f} · cost {cfg['cost_bps_per_side']} bps/side</p>
  <div class="cards">{cards_html}</div>
  <div class="legend-row">
    <span><b>Exit reasons:</b></span>
    {exits_html}
  </div>
</section>

<section id="equity">
  <h2>Equity Curve</h2>
  <p class="subtitle">Strategy vs Buy &amp; Hold (log scale)</p>
  <div id="equity_chart" style="height:420px"></div>
</section>

<section id="drawdown">
  <h2>Drawdown</h2>
  <p class="subtitle">Strategy and Buy &amp; Hold drawdown from running peak</p>
  <div id="dd_chart" style="height:300px"></div>
</section>

<section id="signals">
  <h2>Buy/Sell Timing &amp; Exit Reasons</h2>
  <p class="subtitle">▲ entry &nbsp; ▼ exit (color = reason). Hover for details.</p>
  <div id="signals_chart" style="height:520px"></div>
</section>

<section id="predictions">
  <h2>Predictions</h2>
  <p class="subtitle">Predicted vs actual 5-day forward log return</p>
  <div class="grid-2">
    <div id="pred_scatter" style="height:380px"></div>
    <div id="pred_ts" style="height:380px"></div>
  </div>
</section>

{model_html}

{method_html}

{per_model_html}

{tuning_html}

{('''<section id="sign">
  <h2>Adaptive Sign Correction (trailing OOS IC)</h2>
  <p class="subtitle">Per-bar trailing IC over the last 120 valid (raw_pred, actual) pairs — sign flip when |IC| ≥ 0.04. Solid line = active sign (−1, 0, +1), dashed = trailing IC.</p>
  <div id="sign_chart" style="height:420px"></div>
</section>''') if has_sign_history else ''}

{(f'''<section id="importance">
  <h2>Feature Importance{' &amp; Stability' if imp_stability else ''}</h2>
  <p class="subtitle">{'Mean importance across ' + str(data['refits']) + ' refits · error bars = ±1σ · color = top-10 frequency' if imp_stability else 'Top features by gain (best base model from latest refit)'}</p>
  <div id="imp_chart" style="height:760px"></div>
</section>''') if has_importance else ''}

<section id="trades">
  <h2>Trade Log ({m['n_trades']} trades)</h2>
  <p class="subtitle">Click column header to sort.</p>
  <div class="table-wrap">
    <table class="trades" id="trade_table">
      <thead>
        <tr>
          <th data-key="entry_date">Entry</th>
          <th data-key="exit_date">Exit</th>
          <th data-key="entry_price">Entry $</th>
          <th data-key="exit_price">Exit $</th>
          <th data-key="bars_held">Bars</th>
          <th data-key="pred_at_entry">Pred</th>
          <th data-key="net_return">Net Ret</th>
          <th data-key="exit_reason">Reason</th>
        </tr>
      </thead>
      <tbody id="trade_tbody"></tbody>
    </table>
  </div>
</section>

<script>
const DATA = {payload};
const EXIT_COLORS = {json.dumps(EXIT_COLORS)};

const PLOTLY_LAYOUT = {{
  paper_bgcolor: '#161b22',
  plot_bgcolor:  '#161b22',
  font: {{ color: '#c9d1d9', family: 'Helvetica, Arial, sans-serif', size: 12 }},
  xaxis: {{ gridcolor: '#2a313c', zerolinecolor: '#2a313c' }},
  yaxis: {{ gridcolor: '#2a313c', zerolinecolor: '#2a313c' }},
  margin: {{ l: 60, r: 30, t: 20, b: 40 }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)' }},
}};

const PLOTLY_CONFIG = {{ displaylogo: false, responsive: true }};

function deepMerge(a, b) {{
  const out = JSON.parse(JSON.stringify(a));
  for (const k in b) {{
    if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]))
      out[k] = deepMerge(out[k] || {{}}, b[k]);
    else out[k] = b[k];
  }}
  return out;
}}

// ---------- Equity curve ----------
(function() {{
  const eq = DATA.equity_curve;
  const dates = eq.map(d => d.date);
  const traces = [
    {{ x: dates, y: eq.map(d => d.strategy), name: 'Strategy',
       type: 'scatter', mode: 'lines', line: {{ color: '#58a6ff', width: 2 }} }},
    {{ x: dates, y: eq.map(d => d.buy_hold), name: 'Buy & Hold',
       type: 'scatter', mode: 'lines', line: {{ color: '#7d8590', width: 1.5, dash: 'dot' }} }},
  ];
  const layout = deepMerge(PLOTLY_LAYOUT, {{
    yaxis: {{ type: 'log', title: 'Equity (USD)' }},
    hovermode: 'x unified',
    margin: {{ t: 20 }},
  }});
  Plotly.newPlot('equity_chart', traces, layout, PLOTLY_CONFIG);
}})();

// ---------- Drawdown ----------
(function() {{
  const eq = DATA.equity_curve;
  const dates = eq.map(d => d.date);
  // strategy drawdown
  let peak = -Infinity;
  const stratDD = eq.map(d => {{ peak = Math.max(peak, d.strategy); return (d.strategy/peak - 1)*100; }});
  let bhPeak = -Infinity;
  const bhDD = eq.map(d => {{ bhPeak = Math.max(bhPeak, d.buy_hold); return (d.buy_hold/bhPeak - 1)*100; }});
  const traces = [
    {{ x: dates, y: stratDD, name: 'Strategy',
       fill: 'tozeroy', type: 'scatter', mode: 'lines',
       line: {{ color: '#dc2626', width: 1 }},
       fillcolor: 'rgba(220,38,38,0.25)' }},
    {{ x: dates, y: bhDD, name: 'Buy & Hold',
       type: 'scatter', mode: 'lines',
       line: {{ color: '#7d8590', width: 1, dash: 'dot' }} }},
  ];
  const layout = deepMerge(PLOTLY_LAYOUT, {{
    yaxis: {{ title: 'Drawdown (%)', ticksuffix: '%' }},
    hovermode: 'x unified',
  }});
  Plotly.newPlot('dd_chart', traces, layout, PLOTLY_CONFIG);
}})();

// ---------- Signals overlay on price ----------
(function() {{
  const preds = DATA.predictions;
  const trades = DATA.trades;
  const equity = DATA.equity_curve;
  const dates = preds.map(p => p.date);
  const closes = preds.map(p => p.close);

  // Regime-blocked shapes: shade contiguous non-tradable spans
  const shapes = [];
  if (equity.length && equity[0].tradable !== undefined) {{
    let i = 0;
    while (i < equity.length) {{
      if (equity[i].tradable === 0) {{
        const start = equity[i].date;
        while (i < equity.length && equity[i].tradable === 0) i++;
        const end = (i < equity.length ? equity[i].date : equity[equity.length-1].date);
        shapes.push({{
          type: 'rect', xref: 'x', yref: 'paper',
          x0: start, x1: end, y0: 0, y1: 1,
          fillcolor: 'rgba(217,119,6,0.10)',
          line: {{ width: 0 }}, layer: 'below',
        }});
      }} else {{ i++; }}
    }}
  }}

  const priceLine = {{
    x: dates, y: closes, name: 'BTC Close',
    type: 'scatter', mode: 'lines',
    line: {{ color: '#58a6ff', width: 1.5 }},
  }};

  // entries: ▲
  const entries = {{
    x: trades.map(t => t.entry_date),
    y: trades.map(t => t.entry_price),
    name: 'Entry', mode: 'markers', type: 'scatter',
    marker: {{ symbol: 'triangle-up', size: 11, color: '#16a34a',
              line: {{ color: '#fff', width: 0.5 }} }},
    text: trades.map(t => `Entry · pred=${{(t.pred_at_entry*100).toFixed(2)}}%`),
    hoverinfo: 'x+y+text',
  }};

  // exits, grouped by reason for separate legend entries
  const reasons = ['take_profit', 'stop_loss', 'time_stop', 'signal_flip'];
  const exitTraces = reasons.map(r => {{
    const sub = trades.filter(t => t.exit_reason === r);
    return {{
      x: sub.map(t => t.exit_date),
      y: sub.map(t => t.exit_price),
      name: 'Exit · ' + r.replace('_',' '),
      mode: 'markers', type: 'scatter',
      marker: {{ symbol: 'triangle-down', size: 10,
                color: EXIT_COLORS[r],
                line: {{ color: '#fff', width: 0.5 }} }},
      text: sub.map(t => `Exit · ${{r}} · ret=${{(t.net_return*100).toFixed(2)}}%`),
      hoverinfo: 'x+y+text',
    }};
  }});

  const layout = deepMerge(PLOTLY_LAYOUT, {{
    yaxis: {{ title: 'BTC Price (USD)', type: 'log' }},
    hovermode: 'closest',
    shapes: shapes,
    annotations: shapes.length ? [{{
      x: 0.01, y: 0.98, xref: 'paper', yref: 'paper', xanchor: 'left',
      text: '<span style="background:rgba(217,119,6,0.15);padding:2px 6px">orange = regime gate blocks new entries</span>',
      showarrow: false, font: {{ size: 11, color: '#d97706' }},
    }}] : [],
  }});
  Plotly.newPlot('signals_chart', [priceLine, entries, ...exitTraces], layout, PLOTLY_CONFIG);
}})();

// ---------- Predictions: scatter + time series ----------
(function() {{
  const preds = DATA.predictions.filter(p => p.predicted_5d !== null && p.actual_5d !== null);
  const px = preds.map(p => p.predicted_5d * 100);
  const py = preds.map(p => p.actual_5d * 100);

  // simple OLS for guide line
  const n = px.length;
  const mx = px.reduce((a,b)=>a+b,0)/n;
  const my = py.reduce((a,b)=>a+b,0)/n;
  let num=0, den=0;
  for (let i=0;i<n;i++) {{ num += (px[i]-mx)*(py[i]-my); den += (px[i]-mx)**2; }}
  const slope = den ? num/den : 0;
  const intercept = my - slope*mx;
  const xRange = [Math.min(...px), Math.max(...px)];

  const scatter = [
    {{ x: px, y: py, mode: 'markers', type: 'scatter',
       marker: {{ color: py.map(v => v >= 0 ? '#16a34a' : '#dc2626'),
                 size: 5, opacity: 0.6 }},
       name: 'samples',
       hovertemplate: 'pred %{{x:.2f}}%<br>actual %{{y:.2f}}%<extra></extra>' }},
    {{ x: xRange, y: xRange.map(v => slope*v + intercept),
       mode: 'lines', name: `OLS slope=${{slope.toFixed(2)}}`,
       line: {{ color: '#d97706', width: 1.5, dash: 'dash' }} }},
  ];
  Plotly.newPlot('pred_scatter', scatter, deepMerge(PLOTLY_LAYOUT, {{
    xaxis: {{ title: 'Predicted 5d return (%)' }},
    yaxis: {{ title: 'Actual 5d return (%)' }},
  }}), PLOTLY_CONFIG);

  const allPreds = DATA.predictions;
  const tsTraces = [
    {{ x: allPreds.map(p => p.date),
       y: allPreds.map(p => p.predicted_5d === null ? null : p.predicted_5d*100),
       name: 'Predicted', type: 'scatter', mode: 'lines',
       line: {{ color: '#58a6ff', width: 1.2 }} }},
    {{ x: allPreds.map(p => p.date),
       y: allPreds.map(p => p.actual_5d === null ? null : p.actual_5d*100),
       name: 'Actual', type: 'scatter', mode: 'lines',
       line: {{ color: '#7d8590', width: 1, dash: 'dot' }}, opacity: 0.6 }},
  ];
  Plotly.newPlot('pred_ts', tsTraces, deepMerge(PLOTLY_LAYOUT, {{
    yaxis: {{ title: 'Return (%)', ticksuffix: '%' }},
    hovermode: 'x unified',
  }}), PLOTLY_CONFIG);
}})();

// ---------- Feature importance with stability ----------
(function() {{
  if (!document.getElementById('imp_chart')) return;
  const stab = (DATA.importance_stability || []).slice(0, 30);
  if (!stab.length) {{
    const fi = DATA.feature_importance.slice(0, 30).reverse();
    Plotly.newPlot('imp_chart', [{{
      x: fi.map(d => d.importance), y: fi.map(d => d.feature),
      type: 'bar', orientation: 'h', marker: {{ color: '#58a6ff' }},
    }}], deepMerge(PLOTLY_LAYOUT, {{
      xaxis: {{ title: 'Importance' }},
      margin: {{ l: 200, r: 30, t: 20, b: 40 }},
    }}), PLOTLY_CONFIG);
    return;
  }}
  const reversed = stab.slice().reverse();
  const refits = DATA.refits || 1;
  const freqKey = Object.keys(reversed[0]).find(k => k.startsWith('top') && k.endsWith('_freq'));
  const trace = {{
    x: reversed.map(d => d.mean_importance),
    y: reversed.map(d => d.feature),
    type: 'bar', orientation: 'h',
    error_x: {{
      type: 'data', symmetric: true,
      array: reversed.map(d => d.std_importance),
      color: '#7d8590', thickness: 1, width: 3,
    }},
    marker: {{
      color: reversed.map(d => d[freqKey] / refits),
      colorscale: [[0, '#1c232e'], [0.5, '#58a6ff'], [1, '#16a34a']],
      cmin: 0, cmax: 1,
      colorbar: {{
        title: {{ text: 'top-10 freq', font: {{ size: 11 }} }},
        thickness: 12, len: 0.5, x: 1.02,
      }},
    }},
    hovertemplate: '<b>%{{y}}</b><br>mean: %{{x:.4f}}<br>±σ: %{{error_x.array:.4f}}<extra></extra>',
  }};
  const layout = deepMerge(PLOTLY_LAYOUT, {{
    xaxis: {{ title: 'Mean importance (gain) ± std across refits' }},
    yaxis: {{ automargin: true }},
    margin: {{ l: 220, r: 80, t: 20, b: 40 }},
  }});
  Plotly.newPlot('imp_chart', [trace], layout, PLOTLY_CONFIG);
}})();

// ---------- Sign correction history ----------
(function() {{
  const el = document.getElementById('sign_chart');
  if (!el) return;
  const sh = DATA.sign_history || [];
  if (!sh.length) {{
    el.innerHTML = '<p style="color:var(--muted);padding:20px">No sign-correction history (adaptive sign disabled).</p>';
    return;
  }}
  const dates = sh.map(r => r.date);
  const modelKeys = Object.keys(sh[0]).filter(k => k.endsWith('_sgn'))
    .map(k => k.replace('_sgn', ''));
  const palette = ['#58a6ff','#16a34a','#dc2626','#d97706','#7c3aed','#22d3ee'];
  const traces = [];
  modelKeys.forEach((mk, i) => {{
    const c = palette[i % palette.length];
    traces.push({{
      x: dates, y: sh.map(r => r[mk + '_sgn']),
      name: mk + ' sign',
      type: 'scatter', mode: 'lines',
      line: {{ color: c, width: 2, shape: 'hv' }},
    }});
    traces.push({{
      x: dates, y: sh.map(r => r[mk + '_ic']),
      name: mk + ' trailing IC',
      type: 'scatter', mode: 'lines',
      line: {{ color: c, width: 1, dash: 'dot' }},
      yaxis: 'y2',
      visible: 'legendonly',
    }});
  }});
  const layout = deepMerge(PLOTLY_LAYOUT, {{
    yaxis: {{ title: 'sign (-1/0/+1)', range: [-1.2, 1.2] }},
    yaxis2: {{ title: 'trailing IC', overlaying: 'y', side: 'right',
              gridcolor: 'rgba(0,0,0,0)' }},
    hovermode: 'x unified',
    legend: {{ orientation: 'h', y: -0.15 }},
  }});
  Plotly.newPlot('sign_chart', traces, layout, PLOTLY_CONFIG);
}})();

// ---------- Trade table ----------
(function() {{
  const tbody = document.getElementById('trade_tbody');
  const trades = DATA.trades.slice();
  let sortKey = 'entry_date';
  let sortDir = 1;

  function render() {{
    trades.sort((a,b) => {{
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === 'number') return (av-bv)*sortDir;
      return String(av).localeCompare(String(bv))*sortDir;
    }});
    tbody.innerHTML = trades.map(t => {{
      const cls = t.net_return >= 0 ? 'pos' : 'neg';
      return `<tr>
        <td>${{t.entry_date}}</td>
        <td>${{t.exit_date}}</td>
        <td>${{t.entry_price.toFixed(2)}}</td>
        <td>${{t.exit_price.toFixed(2)}}</td>
        <td>${{t.bars_held}}</td>
        <td>${{(t.pred_at_entry*100).toFixed(2)}}%</td>
        <td class="${{cls}}">${{(t.net_return*100).toFixed(2)}}%</td>
        <td><span class="pill ${{t.exit_reason}}">${{t.exit_reason.replace('_',' ')}}</span></td>
      </tr>`;
    }}).join('');
  }}
  document.querySelectorAll('#trade_table th').forEach(th => {{
    th.onclick = () => {{
      const k = th.dataset.key;
      if (sortKey === k) sortDir = -sortDir; else {{ sortKey = k; sortDir = 1; }}
      render();
    }};
  }});
  render();
}})();

</script>
</body>
</html>"""

    return HTML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="btc_data/backtest_results.json")
    parser.add_argument("--out", default="dashboard/btc_xgb.html")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    html_str = build_html(data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")

    print(f"Saved -> {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"Open it directly in a browser: file:///{out_path.resolve().as_posix()}")


if __name__ == "__main__":
    main()
