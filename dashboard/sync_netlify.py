"""Sync drag-drop-deployable dashboards into dashboard/netlify and netlify_secure.

Public (no password, self-contained inline data) → dashboard/netlify/
Password-gated (encrypted/secured)              → dashboard/netlify_secure/

Each target folder also gets an index.html landing page that links to the
bundled dashboards — useful when a Netlify deploy shows only filenames.

Templates that fetch external JSON (backtest.html, combined.html, index.html,
predictions.html, sanity.html) are NOT copied: they would 404 once dropped
without their *_data.json sibling. Use the bundled `index_*.html` versions
instead.

Run after rebuilding dashboards:
    ./venv/Scripts/python -m dashboard.sync_netlify
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PUBLIC = [
    "index_backtest.html",
    "index_combined.html",
    "index_predictions.html",
    "index_sanity.html",
    "limit_continuation.html",
    "delist.html",
]

SECURE = [
    "index_backtest_secure.html",
    "index_combined_secure.html",
    "index_predictions_secure.html",
    "index_sanity_secure.html",
    "index_secure.html",
]


LANDING_CSS = """body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;background:#0e1117;color:#c9d1d9;margin:0;padding:48px;line-height:1.6}
h1{font-size:24px;margin:0 0 4px;font-weight:600}
p.sub{color:#7d8590;font-size:13px;margin:0 0 24px}
ul{list-style:none;padding:0;max-width:760px}
li{background:#161b22;border:1px solid #2a313c;border-left:3px solid #58a6ff;border-radius:6px;margin:0 0 10px;padding:14px 18px}
li.secure{border-left-color:#d29922}
a{color:#58a6ff;text-decoration:none;font-weight:600;font-size:15px}
a:hover{text-decoration:underline}
.size{color:#7d8590;font-size:11px;margin-left:8px}
.note{color:#7d8590;font-size:12px;margin-top:24px;max-width:760px}"""


def landing_page(title: str, items: list[tuple[str, str, int, bool]],
                 footer: str = "") -> str:
    """Build a simple landing page listing each bundled dashboard."""
    rows = []
    for fname, label, size, is_secure in items:
        cls = ' class="secure"' if is_secure else ""
        kb = f"{size/1024:.0f} KB" if size < 1024 * 1024 else f"{size/(1024*1024):.1f} MB"
        rows.append(
            f'<li{cls}><a href="{fname}">{label}</a>'
            f'<span class="size">· {fname} · {kb}</span></li>'
        )
    body = "\n".join(rows)
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{LANDING_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="sub">Last synced {when}</p>
<ul>
{body}
</ul>
{f'<p class="note">{footer}</p>' if footer else ''}
</body>
</html>"""


# Friendly labels for the landing-page links
LABELS = {
    "index_backtest.html":             "回测仪表盘 (Backtest)",
    "index_combined.html":             "综合仪表盘 (OOS + 回测 + 策略)",
    "index_predictions.html":          "预测仪表盘 (Live Top-N + 特征图)",
    "index_sanity.html":               "数据完整性检查 (Sanity)",
    "btc_xgb.html":                    "BTC XGBoost 多模型回测",
    "limit_continuation.html":         "A股连板概率分析",
    "delist.html":                     "A股退市风险预测",
    "index_backtest_secure.html":      "回测仪表盘 (加密版)",
    "index_combined_secure.html":      "综合仪表盘 (加密版)",
    "index_predictions_secure.html":   "预测仪表盘 (加密版)",
    "index_sanity_secure.html":        "数据检查 (加密版)",
    "index_secure.html":               "单页仪表盘 (加密版)",
}


def sync(target_dir: Path, file_list: list[str], title: str,
         footer: str, is_secure: bool, dry_run: bool = False) -> None:
    items = []
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n→ {target_dir.name}/  ({len(file_list)} files)")
    for fname in file_list:
        src = ROOT / fname
        if not src.exists():
            print(f"  [skip] {fname} — not found")
            continue
        size = src.stat().st_size
        items.append((fname, LABELS.get(fname, fname), size, is_secure))
        if not dry_run:
            shutil.copy2(src, target_dir / fname)
        print(f"  [ok] {fname}  ({size/1024:.0f} KB)")

    if items and not dry_run:
        landing = target_dir / "index.html"
        landing.write_text(landing_page(title, items, footer), encoding="utf-8")
        print(f"  [ok] index.html  (landing page)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="list what would be copied, don't write")
    args = ap.parse_args()

    sync(ROOT / "netlify", PUBLIC,
         title="能工智人 · 仪表盘合集 (公开)",
         footer="本目录为可直接拖到 Netlify 部署的公开页面，无需密码。每个 HTML 已内嵌数据。",
         is_secure=False, dry_run=args.dry_run)

    sync(ROOT / "netlify_secure", SECURE,
         title="能工智人 · 仪表盘合集 (加密)",
         footer="本目录的 HTML 需要在页面上输入解密口令才能查看内容。同样支持拖到 Netlify 部署。",
         is_secure=True, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
