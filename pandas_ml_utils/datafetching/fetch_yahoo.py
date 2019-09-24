import logging
import traceback

import cachetools.func
import pandas as pd


def inner_join(df, join: pd.DataFrame, prefix: str = ''):
    return pd.merge(df, join.add_prefix(prefix), left_index=True, right_index=True, how='inner', sort=True)


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def fetch_yahoo(period='max', **kwargs: str):
    import yfinance as yf

    df = None
    for k, v in kwargs.items():
        px = f'{k}_'
        df_ = None

        # bloody skew index does not have any data on yahoo
        if v == '^SKEW':
            df_ = pd.read_csv('http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/skewdailyprices.csv',
                              skiprows=1,
                              parse_dates=True,
                              index_col='Date') \
                .drop(['Unnamed: 2', 'Unnamed: 3'], axis=1)
        else:
            ticker = yf.Ticker(v)
            try:
                # first ty to append the most recent data
                df_ = ticker.history(period="1d", interval='1d')[-1:].combine_first(ticker.history(period=period))
            except:
                traceback.print_exc()
                logging.warning('failed to add yf.Ticker({v}).history(period="1d", interval="1d")[-1:] fallback to hist only!')
                df_ = ticker.history(period=period)

        # print some statistics
        logging.info(f'number of rows for {k} = {len(df_)}, from {df_.index[0]} to {df_.index[-1]} period={period}')

        if df is None:
            df = df_.add_prefix(px)
        else:
            df = df.inner_join(df_, prefix=px)

    # print some statistics
    logging.info(f'number of rows for joined dataframe = {len(df)}, from {df.index[0]} to {df.index[-1]}')
    return df
