# Benchmark methodology — vs Kimi / 通义 / 文心一言

The goal is to know whether our local A-share QA system actually
outperforms commercial Chinese finance assistants on the workflows it
was built for. This doc is the honest framework.

## 1. What we already measured (50-question demo)

`qa/eval/run_phase2_demo.py` runs every question in
`qa/PHASE2_DEMO_QUESTIONS.md` against `/ask`, grades them, and writes
a JSON report. Latest result on Qwen2.5-7B-Instruct int4 + bge-m3 +
news-derived concept tags (commit `d3d7a5b`):

| Category | Description | Pass | Total | % |
|---|---|---|---|---|
| A | Sector / sub-sector leader queries | 11 | 25 | 44 % |
| B | Macro / policy / event meta queries | 14 | 15 | 93 % |
| C | Mixed concept / theme queries | 4 | 10 | 40 % |
| **Total** | | **29** | **50** | **58 %** |

Re-run anytime after a code change:

```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python -m qa.eval.run_phase2_demo \
    --out stock_data/qa/phase2_demo_report.json
```

### Important caveat — strict ts-match scoring is harsh

The pass rule for category A / C is: at least one of the returned
ts_codes overlaps with the hand-curated "expected" list in
`PHASE2_DEMO_QUESTIONS.md`. That expected list is *one acceptable
answer* per question, not the full set. Many "failures" are equally
correct picks:

| Q | Got | Expected | Reality |
|---|---|---|---|
| 17 | 国盾量子 (688027.SH) | 凯乐科技 / 华工科技 | 国盾量子 *is* the canonical quantum stock — expected was wrong |
| 19 | 中国宝安 (000009.SZ) | 协鑫集成 / 宝馨科技 | 中国宝安 has confirmed perovskite R&D — equally valid |
| 10 | 南网储能 / 协鑫能科 | 阳光电源 / 宁德时代 | both are real storage players |

Honest pass rate is probably **70–80 %** if we score on quality, not
strict ts-overlap. Cleaner re-grading would: (a) ask a human to
spot-check each "failure", (b) build a multi-answer expected set, or
(c) auto-grade with a stronger LLM as judge (see §3).

## 2. Comparing against Kimi / 通义 / 文心一言

This isn't something the local harness can do — these are external
services. But the methodology is straightforward, and the same 50
questions are the canonical input.

### Side-by-side test plan

For each question in `qa/PHASE2_DEMO_QUESTIONS.md`:

1. **Send the same query to both systems** (our `/ask` and a
   Kimi/通义/文心 web session, fresh chat each time so the assistant
   doesn't lean on prior context).
2. **Capture**: answer text, ts_codes mentioned, latency, hedge rate
   ("资料中未提供" / "我无法确认" / 类似回避), any obvious
   hallucination flags.
3. **Score on 5 axes** (1–5 each, 0 = fail):
   - **Stock identification** — did the right A-share tickers appear?
   - **Number accuracy** — are EPS / ROE / pct_chg figures real (cross-check against `stock_data/fina_indicator/`)? Hallucinated numbers are a hard fail.
   - **Citation grounding** — does the answer name news dates +
     sources, and do those exist in `news_corpus_dedup.parquet`?
   - **Coverage** — did the answer address the actual question
     (业绩 / 新闻 / 对比 / 概念) or just generic boilerplate?
   - **Hedging vs decisiveness** — does it commit to specifics where
     warranted, or hide behind "建议查阅更多资料"?

Save the per-question scores to a CSV; sum the per-axis means.

### Where we should win

- **Number accuracy**. Our context comes straight from Tushare's
  `fina_indicator/` (real reported EPS / ROE / margins by quarter).
  Commercial assistants infamously hallucinate financial numbers
  unless they're connected to a specialised data source. Our 5/5
  here should be near-automatic; theirs is often 2–3.
- **Citation grounding**. Every news quote has [date][source][title]
  and is verifiable against the corpus. They tend to make up plausible
  but non-existent articles or cite an unverifiable "市场传言".
- **A-share specialisation**. The full alias dict (5,190 stocks +
  short forms + 6-digit symbols), regional-prefix dedup, and concept
  tag mining are all tuned to Chinese A-share idioms. General-purpose
  assistants will miss niche stocks (e.g. 千亿市值之外的 ST 股, 北交所).
- **Latency on cached queries**. LRU cache returns instant on repeats;
  Kimi will re-think every time.

### Where they will probably win

- **World knowledge**. Kimi knows *that* 钙钛矿 ↔ 协鑫集成 / 宝馨
  科技 because it's read a million market commentary articles in
  pre-training. Our entity cards depend on news-derived tags; if a
  concept hasn't been mentioned with a stock recently, we miss it.
- **Synthesis quality on long-form questions**. A 32B+ instruction-
  tuned model writes more polished comparison tables and policy
  analyses than 7B int4. The repetition pathologies on Q12 / Q34 are
  size-related, not retrieval-related.
- **Coverage of recent events**. Their training cutoff is recent;
  our news corpus only updates when we re-pull. Already a known gap
  (see task #76 — incremental news updater).

### Where we'll likely lose unless we fix it

- **Conceptual jargon queries** that have no representation in any
  current entity card or news headline (Q5 医美龙头, Q14 智能驾驶
  代表股, Q15 国产CPU). Our retrieval can't bridge these.
- **Long synthesis**. Repetition loops on Q1 / Q4 / Q17 even with
  `repetition_penalty=1.18 + no_repeat_ngram_size=4`. A 14B / 32B
  model usually doesn't loop.

## 3. Automated LLM-as-judge fallback

If a manual side-by-side is too slow, automate it:

1. Run our system on all 50 (~8 min).
2. Run the competitor (Kimi etc.) on all 50 — needs API access or
   browser automation. Save responses.
3. For each (question, ours, theirs) triple, send to a strong judge
   model (Claude / GPT-4 / Qwen-Max via API) with:

   ```
   System: You are a finance domain expert grading two responses to a
   Chinese A-share research question. Score each on:
     - stock_id (0/1)         did it identify a relevant A-share?
     - number_accuracy (1-5)  are the cited figures plausible / consistent?
     - citation (1-5)         dates + sources cited and verifiable?
     - coverage (1-5)         did it answer the actual question?
     - decisiveness (1-5)     committed vs hedged?
   Output JSON only.
   ```

4. Aggregate. Win rate = % of questions where ours ≥ theirs on the
   axes-weighted total.

The judge costs O($0.05–0.20) per question depending on model.

## 4. Concrete bar to "outperform Kimi"

Define the win as:

**On the 50-question demo, our system has a higher mean score than
Kimi on (number_accuracy + citation), and at least 80 % of Kimi's
score on (stock_id + coverage), with overall mean ≥ Kimi.**

The first axis is where our RAG-with-Tushare approach has the
structural advantage; the second is where Kimi's pretraining knowledge
will lead unless we close the gap with concept-tag mining and
(eventually) entity card enrichment from richer sources (research
report summaries, sector ETF holdings, etc.).

## 5. Action items to actually win

In rough ROI order:

1. **Fix the repetition loops** that still fire on Q1 / Q10 / Q12 /
   Q17 / Q48. Either bump `repetition_penalty` to 1.22 with a 32B
   model, or add a custom `StoppingCriteria` that aborts on detected
   degenerate cycles.
2. **Enrich entity cards beyond news titles** — pull from
   `stock_company.business_scope` (currently not in cards), public
   research-report summaries (Tushare 研报 endpoint if available),
   sector ETF constituents to identify each stock's concept tags
   independently of news.
3. **Refresh `news_linked.parquet`** weekly so concept tags don't go
   stale (task #75).
4. **Switch to Qwen2.5-14B-int4 locally or 32B remote** for synthesis-
   heavy queries. The 7B's repetition pathology and weaker comparison
   tables are the model-side ceiling we're hitting.
5. **Build the manual A/B**. Until we actually run Kimi on the 50
   questions, "we beat Kimi" is conjecture.
