"""
Rolling window split utilities for deeptime.
Thin wrappers — the actual split logic lives in memmap_dataset.RegressionDataWriter.finalize().
"""

from .config import (
    ROLLING_TRAIN_MONTHS, ROLLING_VAL_MONTHS, ROLLING_TEST_MONTHS,
    ROLLING_STEP_MONTHS, INTERLEAVED_TEST_START, PURGE_GAP_DAYS,
)


def get_split_config(config: dict) -> dict:
    """Extract rolling-split parameters from a config dict."""
    return {
        'rolling_train_months':   config.get('rolling_train_months',   ROLLING_TRAIN_MONTHS),
        'rolling_val_months':     config.get('rolling_val_months',     ROLLING_VAL_MONTHS),
        'rolling_test_months':    config.get('rolling_test_months',    ROLLING_TEST_MONTHS),
        'rolling_step_months':    config.get('rolling_step_months',    ROLLING_STEP_MONTHS),
        'interleaved_test_start': config.get('interleaved_test_start', INTERLEAVED_TEST_START),
        'purge_gap_days':         config.get('purge_gap_days',         PURGE_GAP_DAYS),
    }
