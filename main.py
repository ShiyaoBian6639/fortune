from get_original_data import run
from features.news import run as news_run

# run('download') # download stock data
news_run('download')       # Download news for all stocks

# worldwide news and events
from features.events import run as event_run

# Download all data (5 years)
event_run('download')  # Download everything
