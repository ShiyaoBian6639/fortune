# Dashboard Update — 2026-04-25

Summary of the Chinese-localization + backtest-section update applied to
`dashboard/index.html` and `dashboard/build.py`.

## What changed

### 1. Title

- **Before:** `XGBoost Next-Day Return Prediction`
- **After:** `能工智人`

Header subtitle, nav, section headings, card labels, table headers, plot
axes, filter controls, reason tags and footer all translated to simplified
Chinese. `<html lang="zh-CN">`, font stack now prefers `PingFang SC` /
`Microsoft YaHei`.

### 2. Colors — A-share convention (红涨绿跌)

Swapped the up/down color semantics throughout the dashboard to match the
Chinese convention (opposite of Western):

| Direction | Before | After |
|-----------|--------|-------|
| 上涨 / 正向 (up) | `#2ea043` green | **`#da3633` red** |
| 下跌 / 负向 (down) | `#da3633` red | **`#2ea043` green** |

Applied to: IC daily bars, decile bar chart, long-short cumulative curve,
prediction histogram, stock card borders / probability bars / sparklines /
reason-tag backgrounds, group-table heatmap bars, header gradient tint,
all `.good` / `.bad` CSS text classes.

### 3. New section: 策略回测 (Backtest)

Adds a `backtest` section between **Decile Analysis** and **Performance by
Group**. Shows three scenarios via tabbed selector:

| Scenario | 起始资金 | 约束 | CAGR | Sharpe | MDD |
|----------|---------|------|------|--------|-----|
| 理想组合 | 100 万 | 无流动性约束 | **+379.32%** | 9.33 | -25.04% |
| 真实约束 10M | 1000 万 | 5% ADV 上限 | **+51.40%** | 5.67 | -16.36% |
| 真实约束 100M | 1 亿 | 5% ADV 上限 | **+25.38%** | 3.82 | -15.25% |

For each scenario the panel renders:

- 12 summary cards (期末净值, 累计收益, 年化收益, 年化波动, 夏普比率,
  最大回撤, Calmar, 沪深 300 年化, 年化 Alpha, Beta, 信息比率, 基准最大回撤)
- Log-scale equity curve — 策略 vs 沪深 300
- Drawdown curve (red/green flipped)
- 9 trade-stat cards (总交易次数, 胜率, 平均盈利, 平均亏损, 平均收益,
  持仓中位天数, 止盈触发率, 止损触发率, 持有到期率)

### 4. Build pipeline

Added `load_backtest_data()` in `dashboard/build.py`. It reads from
`plots/backtest_xgb_markowitz/`:

- `equity.csv`, `equity_realistic_10m.csv`, `equity_realistic_100m.csv`
- `trades.csv`, `trades_realistic_10m.csv`, `trades_realistic_100m.csv`

Aligns NAV against CSI300 (`stock_data/index/idx_factor_pro/000300_SH.csv`),
recomputes all metrics (Sharpe, α, β, IR, Calmar) directly from NAV for
self-consistency, and downsamples each curve to ~300 points to keep
`data.json` small (final size: 1.78 MB).

## Commands

### Rebuild dashboard data (includes backtest)

```bash
./venv/Scripts/python -m dashboard.build
```

Runs in ~50s. Writes `dashboard/data.json`.

### Serve locally

```bash
./venv/Scripts/python -m http.server -d dashboard 8000
```

Open <http://localhost:8000/>.

### Package as single password-protected file (for Netlify drop)

```bash
./venv/Scripts/python -m dashboard.package_secure -p YOUR_PASSWORD
```

Alternatives:

- Prompt for password interactively:
  `./venv/Scripts/python -m dashboard.package_secure`
- Via env var:
  `DASH_PW=YOUR_PASSWORD ./venv/Scripts/python -m dashboard.package_secure`

Writes a self-contained `dashboard/index_secure.html` (AES-GCM-256 +
PBKDF2-SHA256 200k iterations, Web Crypto decryption in-browser).

### Deploy to Netlify

1. Go to <https://app.netlify.com/drop>.
2. Drag `dashboard/index_secure.html` onto the drop zone.
3. Netlify returns a fresh `https://<random>.netlify.app/` URL.

To update your **existing** site with the same URL instead of getting a new
one, open Sites → your existing site → Deploys, and drop the file onto the
deploys drag-drop area.

## Files touched

```
dashboard/
├── index.html          # full rewrite: Chinese labels, 红涨绿跌 colors, backtest section
├── build.py            # added load_backtest_data() + wired into build() output
└── CHANGES.md          # this file
```

`package_secure.py` unchanged — its two text-hook points
(`fetch('data.json').then(...)` and `renderLiveTable(data);\n});`) are
still present in the rewritten `index.html`, so encryption still works
without edits.

## Known cosmetic detail

The password-login overlay injected by `package_secure.py` is still in
English (`Protected Dashboard / Enter the access password / Unlock`).
If you want that in Chinese too, edit the `LOGIN_OVERLAY` constant in
`dashboard/package_secure.py`.
