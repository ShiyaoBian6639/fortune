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

    # Three demo questions hand-picked to cover the distinct capability
    # axes of the system — each one exercises a different retrieval +
    # context branch the user might not realise is available.
    EXAMPLES = [
        # 1. Fundamentals + news + price summary on a single stock
        '600519.SH 最近一季度业绩怎么样？',
        # 2. Industry-leader / concept resolution (entity-semantic + concept tags)
        '钙钛矿电池标的有哪些？',
        # 3. Forecast block — modelfactory votes + RSI/momentum trend
        '宁德时代未来走势怎样？',
    ]

    # Mobile-friendly layout. Key principles:
    #   1. Single column on phones — gr.Row defaults wrap when items don't
    #      fit, but we further force min-widths so the chat doesn't get
    #      squashed by the dropdown next to it on small screens.
    #   2. Chatbot uses min_height + max_height (instead of fixed height)
    #      so it scales with viewport; on phones it shrinks gracefully.
    #   3. Example buttons get a min-width so they wrap to 1 / 2 / 3 columns
    #      depending on width.
    #   4. Custom CSS adds touch-target sizing + viewport-aware paddings.
    _mobile_css = """
    .gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
    .gradio-container .prose { font-size: 1rem; line-height: 1.55; }
    /* On narrow viewports stack everything vertically */
    @media (max-width: 720px) {
        .gradio-container { padding: 0.5rem !important; }
        .gradio-container .prose h1 { font-size: 1.4rem !important; margin: 0.3rem 0 !important; }
        .gradio-container .prose p  { font-size: 0.9rem !important; }
        .gr-button { min-height: 44px !important; padding: 8px 12px !important; }
        .gr-textbox textarea { font-size: 16px !important; }   /* prevent iOS zoom on focus */
    }
    /* Example-question buttons — wrap nicely on any width */
    .examples-row { gap: 6px !important; }
    .examples-row > * { flex: 1 1 240px !important; min-width: 220px !important; }
    """

    with gr.Blocks(title='能工智人A股深度解析',
                    theme=gr.themes.Soft(),
                    css=_mobile_css) as demo:
        gr.Markdown("# 能工智人A股深度解析")
        gr.Markdown("可问业绩、新闻、对比、趋势。"
                    "**仅供参考，不作为投资建议。股市有风险，投资需谨慎。**")

        # Filter dropdown is now full-width above the chat (was a side
        # column at scale=1 — squashed on phones, wasted on desktop too).
        ts_filter = gr.Dropdown(
            choices=dropdown_choices,
            value=None,
            label="股票筛选 (可选, 输入代码或名称片段即可过滤; 留空表示不限定)",
            interactive=True,
            allow_custom_value=True,
            filterable=True,
        )

        chatbot = gr.Chatbot(height=None, min_height=320, max_height=600,
                              show_label=False)

        with gr.Row():
            msg = gr.Textbox(label="问题",
                              placeholder="例如: 茅台最近一季业绩如何?  (Enter 提交, Shift+Enter 换行)",
                              lines=2, scale=5, min_width=200)
            submit = gr.Button("提交", variant='primary', scale=1, min_width=80)

        gr.Markdown("**示例问题（点击试用）**：")
        with gr.Row(elem_classes=['examples-row']):
            example_btns = [gr.Button(q, size='sm') for q in EXAMPLES]

        clear = gr.Button("清空对话", size='sm')

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

        def submit_example(q, history, ts_filter):
            # Same pipeline as a manual submit: append user msg, stream
            # assistant message in. Don't echo into the textbox — the
            # example button is a one-click "ask".
            history = history + [
                {'role': 'user',      'content': q},
                {'role': 'assistant', 'content': ''},
            ]
            for partial in ask_qa_stream(q, ts_filter):
                history[-1] = {'role': 'assistant', 'content': partial}
                yield history

        msg.submit(respond, [msg, chatbot, ts_filter], [msg, chatbot])
        submit.click(respond, [msg, chatbot, ts_filter], [msg, chatbot])
        for btn, q in zip(example_btns, EXAMPLES):
            btn.click(
                lambda hist, ts, q=q: (yield from submit_example(q, hist, ts)),
                [chatbot, ts_filter], [chatbot],
            )
        clear.click(lambda: [], None, chatbot)

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
