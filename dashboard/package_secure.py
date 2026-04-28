"""
Package the dashboard as a single password-protected HTML file.

Takes ``dashboard/index.html`` + ``dashboard/data.json``, gzips the JSON,
encrypts it with AES-GCM-256 (key = PBKDF2-HMAC-SHA-256(password, salt, 200k
iterations)), and embeds the ciphertext inline in a copy of the HTML. A login
overlay is injected at the top of ``<body>``; on password entry, the browser's
Web Crypto API derives the key and decrypts in-place, then the existing
dashboard JS runs against the inlined data instead of fetching data.json.

The original dashboard's ``fetch('data.json')`` is patched at runtime by
monkey-patching ``window.fetch`` so we don't have to edit ``index.html``.

Output: ``dashboard/index_secure.html`` — self-contained, deploy-as-one-file.

Usage:
    ./venv/Scripts/python -m dashboard.package_secure --password YOURPASS
    ./venv/Scripts/python -m dashboard.package_secure           # prompts

Deploy:
    1. open https://app.netlify.com/drop (sign in, free tier)
    2. drag the single file index_secure.html onto the page
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
HTML_IN  = ROOT / 'index.html'
DATA_IN  = ROOT / 'data.json'
HTML_OUT = ROOT / 'index_secure.html'

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

LOGIN_OVERLAY = r"""
<div id="_secLogin" style="
  position:fixed;inset:0;background:#0e1117;z-index:9999;
  display:flex;align-items:center;justify-content:center;">
  <div style="
    background:#161b22;border:1px solid #2a313c;border-radius:8px;
    padding:40px;max-width:420px;width:100%;font-family:system-ui;color:#c9d1d9;">
    <h2 style="margin:0 0 8px;font-size:20px;">Protected Dashboard</h2>
    <p style="margin:0 0 20px;color:#7d8590;font-size:13px;">
      XGBoost prediction dashboard for 2026-04-24. Enter the access password.
    </p>
    <form id="_secForm" autocomplete="off">
      <input id="_secPw" type="password" placeholder="Password" autofocus
        style="width:100%;background:#0e1117;border:1px solid #2a313c;
               color:#c9d1d9;padding:10px 12px;border-radius:4px;
               font-size:14px;margin-bottom:12px;box-sizing:border-box;">
      <button type="submit" id="_secBtn"
        style="width:100%;background:#1f6feb;color:white;border:none;
               padding:10px 12px;border-radius:4px;font-size:14px;
               cursor:pointer;font-weight:500;">Unlock</button>
    </form>
    <div id="_secErr" style="color:#da3633;margin-top:12px;font-size:13px;height:18px;"></div>
  </div>
</div>

<script id="_secPayload" type="application/octet-stream">__CIPHERTEXT_B64__</script>

<script>
(function () {
  const ITER = __ITER__;
  const b64ToBytes = (s) => Uint8Array.from(atob(s), c => c.charCodeAt(0));

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

  // Intercept the dashboard's fetch('data.json') so we don't have to modify it.
  let _decrypted = null;
  const _origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url);
    if (url && url.endsWith('data.json') && _decrypted != null) {
      return Promise.resolve(new Response(JSON.stringify(_decrypted), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }));
    }
    return _origFetch(input, init);
  };

  const form = document.getElementById('_secForm');
  const pwEl = document.getElementById('_secPw');
  const err  = document.getElementById('_secErr');
  const btn  = document.getElementById('_secBtn');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    err.textContent = '';
    btn.disabled = true; btn.textContent = 'Decrypting…';
    try {
      _decrypted = await decryptPayload(pwEl.value);
      document.getElementById('_secLogin').remove();
      document.getElementById('_secPayload').remove();
      // Now kick off the dashboard: dispatch the DOMContentLoaded event so the
      // already-registered handlers fire. But the fetch() was bound to the top
      // of the script — actually the dashboard script runs immediately on load
      // and will have already called fetch('data.json'), which (before decrypt)
      // returned a 404. So we need to re-run the dashboard's render pipeline.
      // Simpler: just re-invoke fetch now and dispatch.
      if (typeof window.__RENDER_DASHBOARD__ === 'function') {
        window.__RENDER_DASHBOARD__(_decrypted);
      } else {
        // Fallback: reload the dashboard logic by re-executing the script block
        // (it's idempotent against the fetch we just intercepted).
        location.reload();  // shouldn't happen — we export __RENDER_DASHBOARD__
      }
    } catch (ex) {
      console.error(ex);
      err.textContent = 'Wrong password (or corrupted payload).';
      btn.disabled = false; btn.textContent = 'Unlock';
      pwEl.select();
    }
  });
})();
</script>
"""


def patch_html(html: str, ciphertext_b64: str) -> str:
    """
    Insert the login overlay + decrypt shim, and expose the dashboard's
    render pipeline as ``window.__RENDER_DASHBOARD__(data)`` so the overlay
    can call it directly after decryption (no reload needed).
    """
    # 1. Convert the existing ``fetch('data.json').then(r => r.json()).then(data => { ... })``
    #    into a named function we can call with decrypted data.
    needle = "fetch('data.json').then(r => r.json()).then(data => {"
    if needle not in html:
        raise RuntimeError("Unexpected index.html shape — fetch call not found.")
    replacement = (
        "window.__RENDER_DASHBOARD__ = function (data) {\n"
        "  // (rendering pipeline — formerly the fetch().then() body)\n"
    )
    html = html.replace(needle, replacement)
    # Match the closing `});` that terminated the original `.then(data => { ... });`
    # The original source literally ends with `});` at line containing the
    # closing brace. We turn it into `};`.
    # Use reverse search to find the last `});` before the script end.
    close_needle = "  renderLiveTable(data);\n});"
    if close_needle not in html:
        raise RuntimeError("Unexpected index.html shape — fetch close not found.")
    html = html.replace(close_needle, "  renderLiveTable(data);\n};")

    # 2. Inject login overlay + decrypt script just after <body>.
    overlay = (
        LOGIN_OVERLAY
        .replace('__CIPHERTEXT_B64__', ciphertext_b64)
        .replace('__ITER__', str(PBKDF2_ITERATIONS))
    )
    html = html.replace('<body>', '<body>\n' + overlay, 1)

    return html


def main() -> None:
    ap = argparse.ArgumentParser(description='Encrypt dashboard into a single password-protected HTML.')
    ap.add_argument('--password', '-p', help='Password (prompts if omitted; env DASH_PW also accepted)')
    ap.add_argument('--out', default=str(HTML_OUT), help='Output file path')
    args = ap.parse_args()

    pw = args.password or os.environ.get('DASH_PW') or getpass.getpass('Password: ')
    if not pw:
        raise SystemExit('Password is required.')
    if len(pw) < 6:
        print(f'[package_secure] WARNING: password is only {len(pw)} chars; pick something stronger.')

    html = HTML_IN.read_text(encoding='utf-8')
    data = DATA_IN.read_bytes()
    print(f'[package_secure] data.json: {len(data):,} bytes')

    # Gzip at max level — numeric JSON compresses ~4×.
    gz = gzip.compress(data, compresslevel=9)
    print(f'[package_secure] gzipped:   {len(gz):,} bytes ({len(gz)/len(data):.1%})')

    blob = encrypt(gz, pw)
    print(f'[package_secure] encrypted: {len(blob):,} bytes (AES-GCM-256, PBKDF2 {PBKDF2_ITERATIONS:,} iters)')

    b64 = base64.b64encode(blob).decode('ascii')
    html_out = patch_html(html, b64)

    out = Path(args.out)
    out.write_text(html_out, encoding='utf-8')
    print(f'[package_secure] wrote {out}  ({out.stat().st_size / 1024:.1f} KB)')
    print(f'[package_secure] drag {out.name} onto https://app.netlify.com/drop')


if __name__ == '__main__':
    main()
