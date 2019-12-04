
def SMA(series, timeperiod=30):
    return series.rolling(window=timeperiod).mean()