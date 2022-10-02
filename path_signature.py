import numpy as np
import pandas as pd
import esig as ts


def GetWindow(x, h_window=30, f_window=1):
    # First window
    X = np.array(x.iloc[:h_window,]).reshape(1,-1)

    # Append next window
    for i in range(1, len(x)-h_window+1):
        x_i = np.array(x.iloc[i:i+h_window,]).reshape(1,-1)
        X = np.append(X, x_i, axis=0)

    # Cut end not usable in prediction
    rolling_window = (pd.DataFrame(X)).iloc[:-f_window,]
    return rolling_window


def AddTime(X):
    t = np.linspace(0, 1, len(X))
    return np.c_[t, X]


# Lead - Lag Transform
def Lead(X):
    s = X.shape
    x_0 = X[:,0]
    Lead = np.delete(np.repeat(x_0, 2), 0).reshape(-1, 1)

    for j in range(1, s[1]):
        x_j = X[:,j]
        x_j_lead = np.delete(np.repeat(x_j, 2), 0).reshape(-1, 1)
        Lead = np.concatenate((Lead, x_j_lead), axis=1)

    return Lead


def Lag(X):
    s = X.shape
    x_0 = X[:,0]
    Lag = np.delete(np.repeat(x_0,2),-1).reshape(-1,1)

    for j in range(1,s[1]):
        x_j = X[:,j]
        x_j_lag  = np.delete(np.repeat(x_j,2),-1).reshape(-1,1)
        Lag = np.concatenate((Lag,x_j_lag), axis = 1)

    return Lag


def signature(cls, h_window=30, f_window=1, sig_level=4):

    # Normal window features
    X_window = AddTime(GetWindow(cls, h_window=h_window, f_window=f_window))
    X_window = pd.DataFrame(X_window)

    # signature features
    #Consider only area that has at least f_window future prices left
    cls_slice = cls.iloc[0:(len(cls)-(f_window))]
    cls_array = np.array(cls_slice).reshape(-1,1)
    lag = Lag(cls_array)
    lead = Lead(AddTime(cls_array))
    #concatenate everything to get a datastream
    stream = np.concatenate((lead,lag), axis = 1)
    X_sig = [ts.stream2sig(stream[0:2*h_window-1], sig_level)]

    for i in range(1,(len(cls)-(f_window)-(h_window)+1)):
        stream_i = stream[2*i: 2*(i+h_window)-1]
        signature_i = [ts.stream2sig(stream_i, sig_level)]
        X_sig = np.append(X_sig, signature_i, axis=0)

    ad = np.zeros((len(cls)-len(X_sig), X_sig.shape[1]))
    e = np.vstack((ad, X_sig))

    return pd.DataFrame(e)
