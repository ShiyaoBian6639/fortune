"""
Package the BACKTEST dashboard as a single password-protected HTML file.

Takes ``dashboard/backtest.html`` + ``dashboard/backtest_data.json``, gzips
the JSON, encrypts it with AES-GCM-256 (key = PBKDF2-HMAC-SHA-256(password,
salt, 200k iterations)), and embeds the ciphertext inline in a copy of the
HTML. A login overlay is injected at the top of ``<body>``; on password entry,
the browser's Web Crypto API derives the key, decrypts in-place, and resolves
the deferred Promise that the dashboard is already awaiting.

How it integrates with the dashboard
------------------------------------
``backtest.html``'s data-loader:

    function getData() {
      if (window.BACKTEST_DATA) return Promise.resolve(window.BACKTEST_DATA);
      return fetch('backtest_data.json').then(r => r.json());
    }
    getData().then(data => { ...render... });

We pre-set ``window.BACKTEST_DATA`` to an unresolved Promise *before* the
dashboard script runs (overlay is injected at the top of <body>; dashboard
script is at the bottom). The dashboard's ``Promise.resolve(promise)`` collapses
to the same promise. On password unlock, the overlay calls the deferred
``resolve(decryptedJSON)`` and the dashboard's ``.then(data => {...})``
fires with the decrypted payload.

No source patching of ``backtest.html`` is required.

Output: ``dashboard/index_backtest_secure.html`` — self-contained.

Usage:
    ./venv/Scripts/python -m dashboard.package_secure_backtest --password YOURPASS
    ./venv/Scripts/python -m dashboard.package_secure_backtest             # prompts

Deploy:
    1. open https://app.netlify.com/drop (sign in, free tier)
    2. drag the single file index_backtest_secure.html onto the page
    3. Netlify gives you a URL like https://<random>.netlify.app/
"""

from __future__ import annotations

import argparse
import base64
import getpass
import gzip
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

ROOT = Path(__file__).resolve().parent
HTML_IN  = ROOT / 'backtest.html'
DATA_IN  = ROOT / 'backtest_data.json'
HTML_OUT = ROOT / 'index_backtest_secure.html'

PBKDF2_ITERATIONS = 200_000
KEY_LEN  = 32    # AES-256
SALT_LEN = 16
IV_LEN   = 12    # GCM standard


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode('utf-8'))


def encrypt(plain: bytes, password: str) -> bytes:
    """Return: salt(16) || iv(12) || ciphertext || tag(16)."""
    salt = secrets.token_bytes(SALT_LEN)
    iv   = secrets.token_bytes(IV_LEN)
    key  = derive_key(password, salt)
    aes  = AESGCM(key)
    ct   = aes.encrypt(iv, plain, associated_data=None)   # appends 16-byte tag
    return salt + iv + ct


# ─── Inline login + decrypt overlay ─────────────────────────────────────────
# Notes:
# - Runs BEFORE the dashboard script (we inject right after <body> opens;
#   dashboard <script> is at the bottom of <body>).
# - Pre-sets window.BACKTEST_DATA to a Promise. The dashboard's
#   Promise.resolve(window.BACKTEST_DATA) collapses to the same promise and
#   waits. We resolve it with the decrypted JSON after a successful unlock.

LOGIN_OVERLAY = r"""
<div id="_secLogin" style="
  position:fixed;inset:0;background:#0e1117;z-index:9999;
  display:flex;align-items:center;justify-content:center;">
  <div style="
    background:#161b22;border:1px solid #2a313c;border-radius:8px;
    padding:40px;max-width:420px;width:100%;font-family:system-ui,'PingFang SC','Microsoft YaHei',sans-serif;color:#c9d1d9;">
    <h2 style="margin:0 0 8px;font-size:20px;">受保护的回测看板</h2>
    <p style="margin:0 0 20px;color:#7d8590;font-size:13px;">
      XGB 马科维茨纯多头组合回测看板。请输入访问密码。
    </p>
    <form id="_secForm" autocomplete="off">
      <input id="_secPw" type="password" placeholder="Password" autofocus
        style="width:100%;background:#0e1117;border:1px solid #2a313c;
               color:#c9d1d9;padding:10px 12px;border-radius:4px;
               font-size:14px;margin-bottom:12px;box-sizing:border-box;">
      <button type="submit" id="_secBtn"
        style="width:100%;background:#1f6feb;color:white;border:none;
               padding:10px 12px;border-radius:4px;font-size:14px;
               cursor:pointer;font-weight:500;">解锁</button>
    </form>
    <div id="_secErr" style="color:#da3633;margin-top:12px;font-size:13px;height:18px;"></div>
    <div style="color:#7d8590;margin-top:18px;font-size:11px;">
      AES-GCM-256 · PBKDF2-SHA256 · __ITER__ 轮迭代 · 解密在浏览器本地完成
    </div>
  </div>
</div>

<script id="_secPayload" type="application/octet-stream">__CIPHERTEXT_B64__</script>

<script>
(function () {
  const ITER = __ITER__;
  const b64ToBytes = (s) => Uint8Array.from(atob(s), c => c.charCodeAt(0));

  // Pre-set window.BACKTEST_DATA to a Promise the dashboard's getData() will
  // await. We resolve it on successful decrypt.
  let _resolveData, _rejectData;
  window.BACKTEST_DATA = new Promise((res, rej) => {
    _resolveData = res; _rejectData = rej;
  });

  async function decryptPayload(password) {
    const blob = b64ToBytes(document.getElementById('_secPayload').textContent.trim());
    const salt = blob.slice(0, 16);
    const iv   = blob.slice(16, 28);
    const ct   = blob.slice(28);
    const keyMat = await crypto.subtle.importKey(
      'raw', new TextEncoder().encode(password),
      { name: 'PBKDF2' }, false, ['deriveKey']);
    const key = await crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt, iterations: ITER, hash: 'SHA-256' },
      keyMat, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
    const pt = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ct);
    // Plaintext is gzipped JSON → ungzip via DecompressionStream
    const gzipped = new Response(pt).body;
    const ds = gzipped.pipeThrough(new DecompressionStream('gzip'));
    const text = await new Response(ds).text();
    return JSON.parse(text);
  }

  // Wait for DOM to be ready before binding form (login overlay is at the
  // top of <body> so its DOM exists, but we still defer to be safe).
  function bind() {
    const form = document.getElementById('_secForm');
    const pwEl = document.getElementById('_secPw');
    const err  = document.getElementById('_secErr');
    const btn  = document.getElementById('_secBtn');
    if (!form) { setTimeout(bind, 0); return; }
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      err.textContent = '';
      btn.disabled = true; btn.textContent = '解密中…';
      try {
        const data = await decryptPayload(pwEl.value);
        document.getElementById('_secLogin').remove();
        document.getElementById('_secPayload').remove();
        // Resolve the deferred — the dashboard's getData() promise chain will fire.
        _resolveData(data);
      } catch (ex) {
        console.error(ex);
        err.textContent = '密码错误（或加密数据损坏）。';
        btn.disabled = false; btn.textContent = '解锁';
        pwEl.select();
      }
    });
    pwEl.focus();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind, { once: true });
  } else {
    bind();
  }
})();
</script>
"""


def patch_html(html: str, ciphertext_b64: str) -> str:
    """Inject the login overlay + decrypt shim right after <body> opens.

    No other modification of the source HTML — backtest.html's getData()
    already checks window.BACKTEST_DATA, which our shim sets to a Promise.
    """
    overlay = (
        LOGIN_OVERLAY
        .replace('__CIPHERTEXT_B64__', ciphertext_b64)
        .replace('__ITER__', str(PBKDF2_ITERATIONS))
    )
    if '<body>' not in html:
        raise RuntimeError("backtest.html has no <body> tag — cannot inject overlay")
    return html.replace('<body>', '<body>\n' + overlay, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Encrypt the backtest dashboard into a single password-protected HTML.')
    ap.add_argument('--password', '-p',
                    help='Password (prompts if omitted; env DASH_PW also accepted)')
    ap.add_argument('--out', default=str(HTML_OUT), help='Output file path')
    ap.add_argument('--html', default=str(HTML_IN),
                    help=f'Input HTML (default {HTML_IN.name})')
    ap.add_argument('--data', default=str(DATA_IN),
                    help=f'Input JSON (default {DATA_IN.name})')
    args = ap.parse_args()

    pw = args.password or os.environ.get('DASH_PW') or getpass.getpass('Password: ')
    if not pw:
        raise SystemExit('Password is required.')
    if len(pw) < 6:
        print(f'[package_secure_backtest] WARNING: password is only {len(pw)} chars; pick something stronger.')

    html_in = Path(args.html)
    data_in = Path(args.data)
    if not html_in.exists():
        raise SystemExit(f'Missing {html_in} — run `dashboard.build_backtest` first?')
    if not data_in.exists():
        raise SystemExit(f'Missing {data_in} — run `dashboard.build_backtest` first?')

    html = html_in.read_text(encoding='utf-8')
    data = data_in.read_bytes()
    print(f'[package_secure_backtest] {data_in.name}: {len(data):,} bytes')

    # Gzip at max level — JSON compresses well.
    gz = gzip.compress(data, compresslevel=9)
    print(f'[package_secure_backtest] gzipped:   {len(gz):,} bytes ({len(gz)/len(data):.1%})')

    blob = encrypt(gz, pw)
    print(f'[package_secure_backtest] encrypted: {len(blob):,} bytes '
          f'(AES-GCM-256, PBKDF2 {PBKDF2_ITERATIONS:,} iters)')

    b64 = base64.b64encode(blob).decode('ascii')
    html_out = patch_html(html, b64)

    out = Path(args.out)
    out.write_text(html_out, encoding='utf-8')
    print(f'[package_secure_backtest] wrote {out}  ({out.stat().st_size / 1024:.1f} KB)')
    print(f'[package_secure_backtest] drag {out.name} onto https://app.netlify.com/drop')


if __name__ == '__main__':
    main()
