Next Steps to Run

  Fix the 15 missing dates in each dataset:
  from api.moneyflow import run as mf_run
  mf_run('range', start_date='20181024', end_date='20181130')   # fill Oct–Nov 2018

  from api.stk_limit import run as sl_run
  sl_run('range', start_date='20200623', end_date='20200817')   # fill Jun–Aug 2020

  Daily update (run after market close):
  ./venv/Scripts/python update_all.py
  ./venv/Scripts/python update_all.py --skip-stocks   # faster if OHLCV already current

  Download new data sources:
  from api.fina_indicator import run; run('update')    # ~5,190 stocks, takes ~30 min
  from api.block_trade import run; run('download')     # all trading days since 2017

  ▎ Important: The feature count changed from 150 → 218. The existing transformer_classifier.pth checkpoint was trained with the old 150-feature input size and will need to be retrained after downloading the new feature data.

