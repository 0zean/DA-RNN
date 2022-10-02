from get_rolling_window import rolling_window
from itertools import product
import statsmodels.api as sm

def arma(signal):
    best_arima = None
    src = signal

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
            arima = sm.tsa.statespace.SARIMAX(src.values,
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

    return best_arima.predict()


def get_labels_from_features(features, window_size, y_dim):
    return features[window_size - 1:, -y_dim:]


def split_by_ratio(features, validation_ratio):
    length = len(features)
    validation_length = int(validation_ratio * length)

    return features[:-validation_length], features[-validation_length:]


def split_data(data, apply, window_size, y_dim, validation_ratio):
    train_data, val_data = split_by_ratio(data, validation_ratio)

    train_f, train_l = rolling_window(
        train_data, window_size, 1
    ), get_labels_from_features(train_data, window_size, y_dim)

    val_f, val_l = rolling_window(
        val_data, window_size, 1
    ), get_labels_from_features(val_data, window_size, y_dim)

    train_f = train_f[:, :, 0:train_data.shape[1]-1]
    val_f = val_f[:, :, 0:val_data.shape[1]-1]

    return apply(train_f), apply(train_l), apply(val_f), apply(val_l)
