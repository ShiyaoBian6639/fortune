python api/get_data.py --dataset fund_factor_pro --workers 8 --rate 8

# ── First run: process all data and train (full dataset, ~3,752 stocks)
  # --memory_efficient is REQUIRED for full dataset — streams stocks from disk
  # one at a time instead of loading all ~5M sequences into RAM.
  # Builds the 125-feature cache at stock_data/cache/ then trains.
  ./venv/Scripts/python -m dl.main --max_stocks 0 --memory_efficient --epochs 15
  ./venv/Scripts/python -m dl.predict_all
  # ── Subsequent runs: skip reprocessing, load from cache
  ./venv/Scripts/python -m dl.main --use_cache --epochs 50

  # ── Quick smoke-test: 100 stocks, default settings (no --memory_efficient needed)
  ./venv/Scripts/python -m dl.main

  # ── Custom settings
  ./venv/Scripts/python -m dl.main --max_stocks 0 --memory_efficient \
      --epochs 100 --batch_size 512 --learning_rate 1e-4 --loss_type focal

  # ── Predict specific stocks after training
  ./venv/Scripts/python -m dl.main --use_cache --predict_stocks 600000 000001 300750

  Typical workflow:

  Step 1 (one-time, ~30-60 min):
    --max_stocks 0 --memory_efficient
    → processes all 3,752 stocks, builds 125-feature sequences
    → saves to stock_data/cache/ as memory-mapped .npy files

  Step 2 (repeated training experiments):
    --use_cache --epochs 50
    → skips all data processing, loads directly from cache
    → ~2x faster per epoch vs re-processing

  Key flags:

  ┌────────────────────┬─────────────────────────────────────────────────────────────────┐
  │        Flag        │                           When to use                           │
  ├────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ --max_stocks 0     │ Full dataset (all SH + SZ stocks)                               │
  ├────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ --max_stocks 100   │ Default — quick test with 100 stocks                            │
  ├────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ --memory_efficient │ Required for full dataset (>500 stocks); streams from disk      │
  ├────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ --use_cache        │ Skip data processing — reuse stock_data/cache/ from a prior run │
  ├────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ --epochs 50        │ Training epochs (default 50)                                    │
  ├────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ --loss_type focal  │ Focal loss (default) — best for imbalanced 10-class problem     │
  └────────────────────┴─────────────────────────────────────────────────────────────────┘

  Important: If you previously ran with the old 106-feature pipeline, the cache is stale. Delete it before the first full run with the new 125 features:

  rm -rf stock_data/cache/
  ./venv/Scripts/python -m dl.main --max_stocks 0 --memory_efficient

 After cache is already built

  Phase 1 only (frozen BERT, fast):
  ./venv/Scripts/python -m multimodal.main --mode train --phase 1

  Phase 2 only (LoRA fine-tune, requires Phase 1 checkpoint):
  ./venv/Scripts/python -m multimodal.main --mode train --phase 2

  Both phases sequentially (skips preprocessing):
  ./venv/Scripts/python -m multimodal.main --mode train

  Evaluate best checkpoint on test set:
  ./venv/Scripts/python -m multimodal.main --mode evaluate --phase 1
  ./venv/Scripts/python -m multimodal.main --mode evaluate --phase 2

  ---
  Resuming Phase 2 after manual stop

  Just re-run the same command — it now auto-resumes:
  ./venv/Scripts/python -m multimodal.main --mode train --phase 2

  What happens under the hood:
  - After each epoch, phase2_resume_model.pth + phase2_resume_bert.pth + phase2_resume.json are saved to the checkpoint dir (overwriting previous epoch — only one resume slot)
  - On restart, it reads phase2_resume.json, loads the weights, and continues from epoch N+1 with the preserved best_val_f1 and patience_ctr
  - When Phase 2 finishes normally, resume files are deleted so the next run starts fresh

  To force a fresh Phase 2 run (discard previous progress):
  rm stock_data/checkpoints/multimodal/phase2_resume*.pth stock_data/checkpoints/multimodal/phase2_resume.json
  ./venv/Scripts/python -m multimodal.main --mode train --phase 2



  Step 1 — Preprocess (build BERT embedding cache + align data)

  ./venv/Scripts/python -m multimodal.main --mode preprocess

  Builds:
  - stock_data/cache/news_embeddings.npz — MacBERT embeddings (Phase 1)
  - stock_data/cache/news_tokens.npz — token cache (Phase 2)
  - stock_data/cache/multimodal/ — aligned price+news dataset

  ---
  Step 2 — Train Phase 1 (frozen BERT)

  ./venv/Scripts/python -m multimodal.main --mode train --phase 1

  ---
  Step 3 — Train Phase 2 (LoRA fine-tune BERT)

  ./venv/Scripts/python -m multimodal.main --mode train --phase 2

  ---
  Step 4 — Evaluate

  ./venv/Scripts/python -m multimodal.main --mode evaluate --phase 2

  ---
  Or run everything in one command

  ./venv/Scripts/python -m multimodal.main --mode all

  ---
  Quick smoke test (5 stocks, 3 news days)

  ./venv/Scripts/python -m multimodal.main --mode all --max_stocks 5 --max_days 3

  All commands must be run from D:\didi\stock\tushare.

./venv/Scripts/python -m multimodal.main --mode preprocess
./venv/Scripts/python -m multimodal.main --mode train --phase 1
./venv/Scripts/python -m multimodal.main --mode train --phase 2

# Step 1 — build news embedding + token caches (one-time, ~10 min on GPU)
  ./venv/Scripts/python -m multimodal.main --mode preprocess

  # Step 2 — Phase 1 training (frozen BERT, fast, ~few hours)
  ./venv/Scripts/python -m multimodal.main --mode train --phase 1

  # Step 3 — Phase 2 training (LoRA fine-tune, slower)
  ./venv/Scripts/python -m multimodal.main --mode train --phase 2

  # Step 4 — Evaluate on test set (prints classification report)
  ./venv/Scripts/python -m multimodal.main --mode evaluate --phase 2

  Or run everything in one command:

  ./venv/Scripts/python -m multimodal.main --mode all

  evaluate runs the best Phase 2 checkpoint on the held-out test set and prints per-class accuracy, F1, and a confusion matrix.

   Full production run (all stocks, Phase 2 checkpoint):
  ./venv/Scripts/python -m multimodal.main --mode predict

  Specify phase (use Phase 1 checkpoint instead):
  ./venv/Scripts/python -m multimodal.main --mode predict --phase 1

  Custom output path:
  ./venv/Scripts/python -m multimodal.main --mode predict --output /path/to/my_predictions.csv

  Default output location: stock_data/predictions/predictions_YYYYMMDD.csv