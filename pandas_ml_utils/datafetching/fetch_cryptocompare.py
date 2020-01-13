import datetime

import cachetools
import pandas as pd
from pandas_ml_utils.extern.cryptocompare import CURR, LIMIT, TIME, get_historical_price_day, get_historical_price_hour
# TODO need to be implemented analog fetch yahoo


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def fetch_cryptocompare_daily(coin, curr=CURR, limit=LIMIT) -> pd.DataFrame:
    data = get_historical_price_day(coin, curr, limit)
    return _data_to_frame(data)


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def fetch_cryptocompare_hourly(coin, curr=CURR, limit=LIMIT) -> pd.DataFrame:
    data = get_historical_price_hour(coin, curr, limit)
    return _data_to_frame(data)


def _data_to_frame(data):
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex(df[TIME].apply(datetime.datetime.fromtimestamp))
    return df.drop(TIME)