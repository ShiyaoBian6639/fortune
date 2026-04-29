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

    def ask_qa(question, history, ts_filter):
        # If a stock filter is selected, prepend the ts_code to the query
        if ts_filter and ts_filter.strip():
            ts = ts_filter.split()[0]
            if ts not in question:
                question = f"{ts} {question}"
        try:
            r = requests.post(f"{args.api}/ask",
                               json={'query': question, 'top_k': 5},
                               timeout=120)
            r.raise_for_status()
            data = r.json()
            answer = data.get('answer', '(no answer)')
            ts_codes = data.get('ts_codes', [])
            n_arts = data.get('n_articles', 0)
            elapsed = data.get('elapsed_seconds', 0)
            footer = (f"\n\n---\n*resolved: {ts_codes}　"
                      f"articles: {n_arts}　"
                      f"latency: {elapsed:.1f}s*")
            return answer + footer
        except Exception as e:
            return f"❌ error: {e}"

    with gr.Blocks(title='无能囃人A股深度解析') as demo:
        gr.Markdown("# 无能囃人A股深度解析")
        gr.Markdown("基于 tushare 全量数据 + Qwen2.5-7B + RAG。"
                    "可问业绩、新闻、对比等。")
        with gr.Row():
            ts_filter = gr.Dropdown(
                choices=dropdown_choices,
                label="股票筛选 (可选, 输入代码或名称片段即可过滤)",
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
                return '', history
            answer = ask_qa(msg, history, ts_filter)
            history = history + [
                {'role': 'user',      'content': msg},
                {'role': 'assistant', 'content': answer},
            ]
            return '', history

        msg.submit(respond, [msg, chatbot, ts_filter], [msg, chatbot])
        submit.click(respond, [msg, chatbot, ts_filter], [msg, chatbot])
        clear.click(lambda: [], None, chatbot)

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
