"""
CLI driver for `model_compare`.

Usage:
    # Quick smoke test: 5 folds on lightgbm
    ./venv/Scripts/python -m model_compare.main --engine lightgbm --max_folds 5

    # Full walk-forward (212 folds) on a single engine
    ./venv/Scripts/python -m model_compare.main --engine xgb_shallow

    # Run multiple engines sequentially
    ./venv/Scripts/python -m model_compare.main --engines xgb_default xgb_shallow xgb_deep

The walk-forward CV reuses the panel + folds from `xgbmodel`. Output goes to
`stock_data/models_<engine>/xgb_preds/test.csv` so the existing backtest +
dashboard infra can consume each engine's predictions unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from xgbmodel.config      import get_config
from xgbmodel.data_loader import build_panel

from model_compare.walk        import run_walk_forward
from model_compare.engines_gbm import ENGINES as GBM_ENGINES

# Lazy NN imports (torch import is slow; only load if requested)
def _load_nn_engines():
    try:
        from model_compare.transformer_reg import TransformerEngine
        from model_compare.tft             import TFTEngine
        return {'transformer_reg': TransformerEngine, 'tft': TFTEngine}
    except Exception as e:
        print(f"[main] NN engines unavailable: {e}")
        return {}

ENGINES = dict(GBM_ENGINES)
ENGINES.update(_load_nn_engines())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', choices=list(ENGINES.keys()), default=None,
                   help='Single engine to run.')
    p.add_argument('--engines', nargs='+', default=None,
                   help='Run multiple engines sequentially.')
    p.add_argument('--max_folds', type=int, default=0, help='0 = all folds')
    p.add_argument('--max_stocks', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--max_depth',     type=int,   default=None)
    p.add_argument('--n_estimators',  type=int,   default=None)
    args = p.parse_args(argv)

    overrides = {
        'max_stocks': args.max_stocks,
        'device':     args.device,
        'max_folds':  args.max_folds,
    }
    cfg = get_config(**overrides)
    if args.learning_rate is not None:
        cfg['xgb_params']['learning_rate'] = args.learning_rate
    if args.max_depth is not None:
        cfg['xgb_params']['max_depth'] = args.max_depth
    if args.n_estimators is not None:
        cfg['xgb_params']['n_estimators'] = args.n_estimators

    engines_to_run = []
    if args.engine:
        engines_to_run.append(args.engine)
    if args.engines:
        engines_to_run.extend(args.engines)
    if not engines_to_run:
        p.print_help()
        print("\nERROR: specify --engine or --engines", file=sys.stderr)
        return 2

    # Build the panel ONCE and reuse across all engines (saves significant
    # I/O when running multiple engines back-to-back)
    print(f"[main] building panel once for {len(engines_to_run)} engine(s)")
    panel = build_panel(cfg)

    for name in engines_to_run:
        engine_cls = ENGINES[name]
        engine     = engine_cls(cfg)
        print(f"\n{'='*72}\n[main] running engine: {name}\n{'='*72}")
        run_walk_forward(engine, cfg, panel=panel)
    return 0


if __name__ == '__main__':
    sys.exit(main())
