import numpy as np
import pandas as pd
import legitindicators as li
import talib
from itertools import product
import statsmodels.api as sm

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

from path_signature import *


# Arima Model
def arma_(signal):
    best_arima = None
    src = signal
    log = np.log(src)
    df = log.diff()
    df_log = pd.DataFrame(df)

    def reverse_close(array):
        return np.exp(array + log.shift(1))

    Qs = range(0, 2)
    qs = range(0, 3)
    Ps = range(0, 3)
    ps = range(0, 3)
    D = 1
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    best_aic = float("inf")
    for first, second, third, fourth in parameters_list:
        try:
            arima = sm.tsa.statespace.SARIMAX(df_log.values,
                                              order=(first, D, second),
                                              seasonal_order=(third, D, fourth, 4)).fit(disp=True,
                                                                                        maxiter=200,
                                                                                        method='powell')
        except:
            continue
        aic = arima.aic
        if aic < best_aic and aic:
            best_arima = arima
            best_aic = aic

    return reverse_close(best_arima.predict())


# Fisher Transform
def FISH(cls, period: int = 10, adjust: bool = True):
    np.seterr(divide="ignore")

    med = cls
    ndaylow = med.rolling(window=period).min()
    ndayhigh = med.rolling(window=period).max()
    raw = (2 * ((med - ndaylow) / (ndayhigh - ndaylow))) - 1
    smooth = raw.ewm(span=5, adjust=adjust).mean()
    _smooth = smooth.fillna(0)

    return pd.Series(
        (np.log((1 + _smooth) / (1 - _smooth))).ewm(span=3, adjust=adjust).mean(),
        name="{0} period FISH.".format(period),)


# Inverse Fisher Transform of RSI
def IFT_RSI(cls, rsi_period: int = 5, wma_period: int = 9,):
    v1 = pd.Series(0.1 * (talib.RSI(cls, timeperiod=rsi_period) - 50), name="v1")
    d = (wma_period * (wma_period + 1)) / 2
    weights = np.arange(1, wma_period + 1)

    def linear(w):
        def _compute(x):
            return (w * x).sum() / d

        return _compute

    _wma = v1.rolling(wma_period, min_periods=wma_period)
    v2 = _wma.apply(linear(weights), raw=True)

    ift = pd.Series(((v2 ** 2 - 1) / (v2 ** 2 + 1)), name="IFT_RSI")

    return ift


# Jurik Moving Average
def JMA(cls, length=7, phase=50, power=2):
    if phase < -100:
        phase_ratio = 0.5
    elif phase > 100:
        phase_ratio = 2.5
    else:
        phase_ratio = (phase / 100 + 1.5)

    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta**power

    e0=np.zeros(len(cls))
    e1=np.zeros(len(cls))
    e2=np.zeros(len(cls))
    jma = np.zeros(len(cls))
    for i in range(len(cls)):
        e0[i] = (1 - alpha) * cls[i] + alpha * e0[i-1]
        e1[i] = (cls[i] - e0[i]) * (1 - beta) + beta * e1[i-1]
        e2[i] = (e0[i] + phase_ratio * e1[i] - jma[i-1]) * ((1 - alpha)**2) + (alpha**2) * e2[i-1]
        jma[i] = e2[i] + jma[i-1]

    return jma


class Data_Generator:
    def __init__(self, series, window_size, y_dim):
        self.s = series
        self.ws = window_size
        self.yd = y_dim

    def feature_generator(self):
        source, n_past, n_future = self.s, self.ws, self.yd


        # TSFRESH Features
        df = source['1. open']
        df_melted = pd.DataFrame({"open": df.copy()})
        df_melted["dates"] = df_melted.index
        df_melted["Symbols"] = "AAPL"

        df_rolled = roll_time_series(df_melted, column_id="Symbols", column_sort="dates",
                                     max_timeshift=20, min_timeshift=5)

        X = extract_features(df_rolled.drop("Symbols", axis=1),
                             column_id="id", column_sort="dates", column_value="open",
                             impute_function=impute, show_warnings=False)

        X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
        X.index.name = "last_date"

        y = df_melted.set_index("dates").sort_index().open.shift(-1)

        y = y[y.index.isin(X.index)]
        X = X[X.index.isin(y.index)]

        X_train = X["2022-11-10 09:40:00":"2022-11-15 15:40:00"]
        y_train = y["2022-11-10 09:40:00":"2022-11-15 15:40:00"]

        X_train_selected = select_features(X_train, y_train)

        X = X[X_train_selected.columns]

        # Index array for TA
        periods = np.double(np.array(list(range(0, len(source['1. open'])))))

        # Technical Analysis functions
        ht = talib.HT_DCPERIOD(source['1. open']) # Hilbert Transform Dominant Cycle Period
        std = talib.STDDEV(source['1. open'], timeperiod=14, nbdev=1) # Standard Deviation
        htm = talib.HT_TRENDMODE(source['1. open']) # Hilbert Transform Trend
        rsi = talib.RSI(source['1. open'], timeperiod=14) # Relative Strenght Index
        wma = talib.WMA(source['1. open'], timeperiod=20) # Weighted Moving Average
        mavp = talib.MAVP(source['1. open'], periods, minperiod=2, maxperiod=30, matype=0) # Moving Average Variable Period
        roc = talib.ROC(source['1. open']) # Rate of Change
        cmo = talib.CMO(source['1. open']) # Chande Momentum Oscillator
        natr = talib.NATR(source['2. high'].shift(1), source['3. low'].shift(1), source['4. close'].shift(1), timeperiod=14) # Norm ATR
        hdp = talib.HT_DCPHASE(source['1. open']) # Hilbert Transform Dominant Cycle Phase
        ang = talib.LINEARREG_ANGLE(source['1. open'], timeperiod=14) # Linear Regression Angle
        one = source['4. close'].shift(1)
        two = source['4. close'].shift(2)
        three = source['4. close'].shift(3)
        four = source['4. close'].shift(4)
        five = source['4. close'].shift(5)
        fish = FISH(source['1. open'])
        oc1 = source['4. close'].shift(1) / source['1. open']
        irsi = IFT_RSI(source['1. open'])
        ss = li.super_smoother(source['1. open'], 20)
        rf = li.roofing_filter(source['1. open'], 20, 50)
        dc = li.decycler(source['1. open'], 20)
        hc = li.hurst_coefficient(source['1. open'], 14)
        sw = li.ebsw(source['1. open'], 20, 20)
        tf = li.trendflex(source['1. open'], 20)
        jma = JMA(cls=source['1. open'])
        ip, qt = talib.HT_PHASOR(source['1. open'])
        kurt = source['1. open'].rolling(14).kurt() # Rolling Kurtosis
        skew = source['1. open'].rolling(14).skew() # Rolling Skew
        quantile = source['1. open'].rolling(14).quantile(.4, interpolation='midpoint') # Rolling Quantile
        sig = signature(source['1. open'])
        ma7 = source['1. open'].rolling(window=7).mean()
        ma21 = source['1. open'].rolling(window=21).mean()
        ema26 = source['1. open'].ewm(span=26).mean()
        ema12 = source['1. open'].ewm(span=12).mean()
        macd = ema12 - ema26
        sd20 = source['1. open'].rolling(20).std()
        ub = ma21 + (sd20*2)
        lb = ma21 - (sd20*2)
        arma = arma_(source['1. open']) # Arima model

        # Append features to open price & append target (close price) at the end
        feats = [ht,std,htm,rsi,wma,mavp,roc,cmo,natr,hdp,ang,one,two,
                 three,four,five,fish,oc1,irsi,ss,rf,dc,hc,sw,tf,jma,ip,qt,kurt,
                 skew,quantile,ma7,ma21,ema26,ema12,macd,sd20,ub,lb,arma]

        x = np.array(source['1. open']).reshape(-1, 1)
        for i in feats:
            x = np.hstack((x, np.array(i).reshape(-1, 1)))
        x = np.hstack((x, np.array(sig))) # Truncated Rough Path Signatures
        x = np.hstack((x, np.array(X)))
        x = np.hstack((x, np.array(source['4. close']).reshape(-1, 1)))

        # Remove all rows with NaN values
        x = x[~np.isnan(x).any(axis=1), :]

        x = x.astype(float)

        return x
