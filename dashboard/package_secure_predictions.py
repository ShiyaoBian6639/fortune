"""
Package the predictions dashboard into a single password-protected HTML file.

Same envelope as the other package_secure_* modules: AES-GCM-256 +
PBKDF2-HMAC-SHA256 200k iterations.
"""
from __future__ import annotations

import argparse, base64, getpass, gzip, os, secrets
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

ROOT = Path(__file__).resolve().parent
HTML_IN = ROOT / 'predictions.html'
DATA_IN = ROOT / 'predictions_data.json'
HTML_OUT = ROOT / 'index_predictions_secure.html'
ITER = 200_000


def derive_key(pw, salt):
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(pw.encode())


def encrypt(plain, pw):
    salt = secrets.token_bytes(16); iv = secrets.token_bytes(12)
    return salt + iv + AESGCM(derive_key(pw, salt)).encrypt(iv, plain, None)


OVERLAY = r"""
<div id="_secLogin" style="position:fixed;inset:0;background:#0e1117;z-index:9999;display:flex;align-items:center;justify-content:center;">
  <div style="background:#161b22;border:1px solid #2a313c;border-radius:8px;padding:40px;max-width:440px;width:100%;font-family:system-ui,'PingFang SC','Microsoft YaHei',sans-serif;color:#c9d1d9;">
    <h2 style="margin:0 0 8px;font-size:20px;">受保护的预测看板</h2>
    <p style="margin:0 0 20px;color:#7d8590;font-size:13px;">请输入访问密码。</p>
    <form id="_secForm" autocomplete="off">
      <input id="_secPw" type="password" placeholder="Password" autofocus style="width:100%;background:#0e1117;border:1px solid #2a313c;color:#c9d1d9;padding:10px 12px;border-radius:4px;font-size:14px;margin-bottom:12px;box-sizing:border-box;">
      <button type="submit" id="_secBtn" style="width:100%;background:#1f6feb;color:white;border:none;padding:10px 12px;border-radius:4px;font-size:14px;cursor:pointer;font-weight:500;">解锁</button>
    </form>
    <div id="_secErr" style="color:#da3633;margin-top:12px;font-size:13px;height:18px;"></div>
    <div style="color:#7d8590;margin-top:18px;font-size:11px;">AES-GCM-256 · PBKDF2 __ITER__ · 浏览器本地解密</div>
  </div>
</div>
<script id="_secPayload" type="application/octet-stream">__CIPHERTEXT_B64__</script>
<script>
(function(){
  const ITER = __ITER__;
  const b64 = (s)=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
  let _resolve;
  window.PREDICTIONS_DATA = new Promise(r=>{_resolve=r;});
  async function decrypt(pw){
    const blob=b64(document.getElementById('_secPayload').textContent.trim());
    const salt=blob.slice(0,16),iv=blob.slice(16,28),ct=blob.slice(28);
    const km=await crypto.subtle.importKey('raw',new TextEncoder().encode(pw),{name:'PBKDF2'},false,['deriveKey']);
    const k=await crypto.subtle.deriveKey({name:'PBKDF2',salt,iterations:ITER,hash:'SHA-256'},km,{name:'AES-GCM',length:256},false,['decrypt']);
    const pt=await crypto.subtle.decrypt({name:'AES-GCM',iv},k,ct);
    return JSON.parse(await new Response((new Response(pt)).body.pipeThrough(new DecompressionStream('gzip'))).text());
  }
  function bind(){
    const f=document.getElementById('_secForm'); if(!f){setTimeout(bind,0);return;}
    f.addEventListener('submit',async e=>{
      e.preventDefault();
      const pw=document.getElementById('_secPw'),btn=document.getElementById('_secBtn'),err=document.getElementById('_secErr');
      err.textContent='';btn.disabled=true;btn.textContent='解密中…';
      try{
        const d=await decrypt(pw.value);
        document.getElementById('_secLogin').remove();
        document.getElementById('_secPayload').remove();
        _resolve(d);
      }catch(ex){console.error(ex);err.textContent='密码错误。';btn.disabled=false;btn.textContent='解锁';pw.select();}
    });
    document.getElementById('_secPw').focus();
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',bind,{once:true});
  else bind();
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--password', '-p')
    args = ap.parse_args()
    pw = args.password or os.environ.get('DASH_PW') or getpass.getpass('Password: ')
    if not pw: raise SystemExit('Password required.')
    html = HTML_IN.read_text(encoding='utf-8')
    data = DATA_IN.read_bytes()
    print(f'data: {len(data):,} bytes')
    gz = gzip.compress(data, compresslevel=9)
    print(f'gzipped: {len(gz):,} ({len(gz)/len(data):.1%})')
    blob = encrypt(gz, pw)
    print(f'encrypted: {len(blob):,}')
    b64s = base64.b64encode(blob).decode()
    overlay = OVERLAY.replace('__CIPHERTEXT_B64__', b64s).replace('__ITER__', str(ITER))
    html_out = html.replace('<body>', '<body>\n' + overlay, 1)
    HTML_OUT.write_text(html_out, encoding='utf-8')
    print(f'wrote {HTML_OUT}  ({HTML_OUT.stat().st_size/1024:.1f} KB)')


if __name__ == '__main__':
    main()
