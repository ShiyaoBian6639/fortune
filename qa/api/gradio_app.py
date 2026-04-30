"""
Gradio chat UI for the stock-research Q&A engine.

Talks to the FastAPI server (default http://127.0.0.1:8080) so the heavy
Qwen model lives in one process. The UI process is light — just an HTTP
client.

Run order:
    1. Start the API server in one terminal:
        ./venv/Scripts/python -m qa.api.server --quant 4bit --port 8080
    2. Start this UI in another:
        ./venv/Scripts/python -m qa.api.gradio_app --api http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import requests

import gradio as gr


def _load_alias_list(api_url: str) -> list:
    try:
        r = requests.get(f'{api_url}/aliases', params={'limit': 5000}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ui] could not load aliases: {e}")
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--api', default='http://127.0.0.1:8080',
                   help='FastAPI base URL')
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--host', default='127.0.0.1',
                   help="Bind address. Use 0.0.0.0 to expose on LAN.")
    p.add_argument('--share', action='store_true',
                   help='Publish a public *.gradio.live tunnel URL (72h expiry).')
    args = p.parse_args()

    aliases = _load_alias_list(args.api)
    # Load all stocks; Gradio's filterable Dropdown narrows the list as the
    # user types either the ts_code, the 6-digit symbol, or any name fragment.
    dropdown_choices = [f"{r['ts_code']}  {r['name']}" for r in aliases]
    print(f"[ui] loaded {len(dropdown_choices):,} stocks for filter dropdown")

    import re as _re
    _TS_CODE_RE = _re.compile(r'^\d{6}\.(SH|SZ)$', _re.IGNORECASE)

    def _maybe_prepend_ts(question, ts_filter):
        if ts_filter and isinstance(ts_filter, str) and ts_filter.strip():
            ts = ts_filter.strip().split()[0]
            if _TS_CODE_RE.match(ts) and ts not in question:
                return f"{ts} {question}"
        return question

    def ask_qa_stream(question, ts_filter):
        """Generator that yields the partial assistant message as tokens
        arrive over SSE. Runs against /ask_stream on the FastAPI server.
        """
        question = _maybe_prepend_ts(question, ts_filter)
        try:
            with requests.post(f"{args.api}/ask_stream",
                                json={'query': question, 'top_k': 5},
                                stream=True, timeout=180) as r:
                r.raise_for_status()
                meta = {}
                buf = ''
                event_type = None
                event_data = []
                for raw in r.iter_lines(decode_unicode=True):
                    # SSE frames are blank-line-separated. Accumulate
                    # within a frame, parse on blank line.
                    if raw is None:
                        continue
                    if raw == '':
                        if not event_data:
                            continue
                        data = '\n'.join(event_data)
                        try:
                            payload = json.loads(data)
                        except Exception:
                            payload = {}
                        if event_type == 'meta':
                            meta = payload
                        elif event_type == 'token':
                            buf += payload.get('text', '')
                            # Stream the partial answer as-is. Gradio
                            # animates the chat bubble; an explicit
                            # "生成中…" suffix would stick around as
                            # garbage if the stream ends mid-line.
                            yield buf
                        elif event_type == 'done':
                            elapsed = payload.get('elapsed_seconds', 0)
                            cached = payload.get('cached', False)
                            footer = (f"\n\n---\n*resolved: {meta.get('ts_codes',[])}　"
                                      f"articles: {meta.get('n_articles',0)}　"
                                      f"latency: {elapsed:.1f}s"
                                      f"{'  (cached)' if cached else ''}*")
                            yield buf + footer
                            return
                        event_type = None
                        event_data = []
                        continue
                    if raw.startswith('event: '):
                        event_type = raw[len('event: '):].strip()
                    elif raw.startswith('data: '):
                        event_data.append(raw[len('data: '):])
        except Exception as e:
            yield f"❌ error: {e}"

    with gr.Blocks(title='无能囃人A股深度解析') as demo:
        gr.Markdown("# 无能囃人A股深度解析")
        gr.Markdown("基于 tushare 全量数据 + Qwen2.5-7B + RAG。"
                    "可问业绩、新闻、对比等。")
        with gr.Row():
            ts_filter = gr.Dropdown(
                choices=dropdown_choices,
                value=None,                # start empty — no auto-selected stock
                label="股票筛选 (可选, 输入代码或名称片段即可过滤; 留空表示不限定)",
                interactive=True,
                allow_custom_value=True,
                filterable=True,
                scale=1,
            )
            chatbot = gr.Chatbot(height=500, scale=3)
        with gr.Row():
            msg = gr.Textbox(label="问题",
                              placeholder="例如: 茅台最近一季业绩如何?  (Enter 提交, Shift+Enter 换行)",
                              lines=2, scale=4)
            submit = gr.Button("提交", variant='primary', scale=1)
        clear = gr.Button("清空")

        def respond(msg, history, ts_filter):
            if not msg or not msg.strip():
                yield '', history
                return
            # Append the user message and an empty assistant message
            # before streaming so the UI shows progress in real time.
            history = history + [
                {'role': 'user',      'content': msg},
                {'role': 'assistant', 'content': ''},
            ]
            for partial in ask_qa_stream(msg, ts_filter):
                history[-1] = {'role': 'assistant', 'content': partial}
                yield '', history

        msg.submit(respond, [msg, chatbot, ts_filter], [msg, chatbot])
        submit.click(respond, [msg, chatbot, ts_filter], [msg, chatbot])
        clear.click(lambda: [], None, chatbot)

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
