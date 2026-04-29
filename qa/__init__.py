"""Chinese stock-research Q&A system over the tushare/ corpus.

Core layout:
  qa/build_alias_dict.py    — stock identity dictionary (ts_code ↔ name ↔ aliases)
  qa/linker/                — stock-news linker (regex + Aho-Corasick MVP)
  qa/rag/                   — retrieval + context assembly + Qwen Q&A engine
  qa/api/                   — FastAPI + Gradio UI for live Q&A
  qa/eval/                  — five-question grounding eval

All artefacts under stock_data/qa/ (gitignored).
"""
