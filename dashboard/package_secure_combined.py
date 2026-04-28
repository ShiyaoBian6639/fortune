"""
Package the COMBINED dashboard (predictions + backtest + Barra) as a single
password-protected HTML file for Netlify drag-drop.

Reads:
  - dashboard/combined.html
  - dashboard/combined_data.json

Writes:
  - dashboard/index_combined_secure.html

Same crypto envelope as package_secure_backtest.py:
  AES-GCM-256, PBKDF2-HMAC-SHA256 200k iterations.
The only difference is the JS resolves window.COMBINED_DATA (not BACKTEST_DATA)
because the unified dashboard's getData() awaits that global instead.

Usage:
    ./venv/Scripts/python -m dashboard.package_secure_combined --password YOURPASS
    ./venv/Scripts/python -m dashboard.package_secure_combined             # prompts
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
HTML_IN  = ROOT / 'combined.html'
DATA_IN  = ROOT / 'combined_data.json'
HTML_OUT = ROOT / 'index_combined_secure.html'

PBKDF2_ITERATIONS = 200_000
KEY_LEN  = 32
SALT_LEN = 16
IV_LEN   = 12


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=KEY_LEN,
        salt=salt, iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode('utf-8'))


def encrypt(plain: bytes, password: str) -> bytes:
    salt = secrets.token_bytes(SALT_LEN)
    iv   = secrets.token_bytes(IV_LEN)
    key  = derive_key(password, salt)
    aes  = AESGCM(key)
    ct   = aes.encrypt(iv, plain, associated_data=None)
    return salt + iv + ct


LOGIN_OVERLAY = r"""
<div id="_secLogin" style="
  position:fixed;inset:0;background:#0e1117;z-index:9999;
  display:flex;align-items:center;justify-content:center;">
  <div style="
    background:#161b22;border:1px solid #2a313c;border-radius:8px;
    padding:40px;max-width:440px;width:100%;
    font-family:system-ui,'PingFang SC','Microsoft YaHei',sans-serif;color:#c9d1d9;">
    <h2 style="margin:0 0 8px;font-size:20px;">受保护的综合看板</h2>
    <p style="margin:0 0 20px;color:#7d8590;font-size:13px;">
      XGB 选股模型 + 马科维茨长多头组合 综合看板。请输入访问密码。
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

  let _resolveData;
  // Combined dashboard's getData() awaits this Promise via Promise.resolve().
  window.COMBINED_DATA = new Promise((res) => { _resolveData = res; });

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
    const gzipped = new Response(pt).body;
    const ds = gzipped.pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(ds).text());
  }

  function bind() {
    const form = document.getElementById('_secForm');
    if (!form) { setTimeout(bind, 0); return; }
    const pwEl = document.getElementById('_secPw');
    const err  = document.getElementById('_secErr');
    const btn  = document.getElementById('_secBtn');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      err.textContent = '';
      btn.disabled = true; btn.textContent = '解密中…';
      try {
        const data = await decryptPayload(pwEl.value);
        document.getElementById('_secLogin').remove();
        document.getElementById('_secPayload').remove();
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
    overlay = (
        LOGIN_OVERLAY
        .replace('__CIPHERTEXT_B64__', ciphertext_b64)
        .replace('__ITER__', str(PBKDF2_ITERATIONS))
    )
    if '<body>' not in html:
        raise RuntimeError("combined.html has no <body> tag — cannot inject overlay")
    return html.replace('<body>', '<body>\n' + overlay, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Encrypt the combined dashboard into a single password-protected HTML.')
    ap.add_argument('--password', '-p',
                    help='Password (prompts if omitted; env DASH_PW also accepted)')
    ap.add_argument('--out', default=str(HTML_OUT), help='Output file path')
    args = ap.parse_args()

    pw = args.password or os.environ.get('DASH_PW') or getpass.getpass('Password: ')
    if not pw:
        raise SystemExit('Password is required.')
    if len(pw) < 6:
        print(f'[package_secure_combined] WARNING: password is only {len(pw)} chars; pick something stronger.')

    if not HTML_IN.exists():
        raise SystemExit(f'Missing {HTML_IN} — run `dashboard.build_combined` first.')
    if not DATA_IN.exists():
        raise SystemExit(f'Missing {DATA_IN} — run `dashboard.build_combined` first.')

    html = HTML_IN.read_text(encoding='utf-8')
    data = DATA_IN.read_bytes()
    print(f'[package_secure_combined] {DATA_IN.name}: {len(data):,} bytes')

    gz = gzip.compress(data, compresslevel=9)
    print(f'[package_secure_combined] gzipped:   {len(gz):,} bytes ({len(gz)/len(data):.1%})')

    blob = encrypt(gz, pw)
    print(f'[package_secure_combined] encrypted: {len(blob):,} bytes '
          f'(AES-GCM-256, PBKDF2 {PBKDF2_ITERATIONS:,} iters)')

    b64 = base64.b64encode(blob).decode('ascii')
    html_out = patch_html(html, b64)

    out = Path(args.out)
    out.write_text(html_out, encoding='utf-8')
    print(f'[package_secure_combined] wrote {out}  ({out.stat().st_size / 1024:.1f} KB)')
    print(f'[package_secure_combined] drag {out.name} onto https://app.netlify.com/drop')


if __name__ == '__main__':
    main()
