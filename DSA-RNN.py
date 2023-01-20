import warnings

import torch

warnings.filterwarnings('ignore')

torch.__version__
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from da_rnn.torch import DARNN, DEVICE
from poutyne import EarlyStopping, Model, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common import *
from feature_generator import *

#%matplotlib qt



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

NORMALIZE_DATA = True
WINDOW_SIZE = 30
Y_DIM = 1
ENCODER_HIDDEN_STATES = 512
DECODER_HIDDEN_STATES = 512
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_RATIO = 0.2
DROPOUT = 0


tseries = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, meta_data = tseries.get_intraday(symbol='AAPL', interval='5min', outputsize='full')
stock = data[::-1]

data_gen = Data_Generator(series=stock, window_size=WINDOW_SIZE, y_dim=Y_DIM)
raw_data = data_gen.feature_generator()

raw_data.shape

scale = StandardScaler().fit(raw_data)

if NORMALIZE_DATA:
    data = scale.transform(raw_data)
else:
    data = raw_data

data


def to_tensor(array):
    return torch.from_numpy(array).float()


train_X, train_y, val_X, val_y = split_data(data, to_tensor, WINDOW_SIZE, Y_DIM, VALIDATION_RATIO)

print('train_X, train_y :', train_X.shape, train_y.shape)
print('  val_X,   val_y :', val_X.shape, val_y.shape)

darnn = DARNN(
    n=train_X.shape[2] - 1,
    T=WINDOW_SIZE,
    m=ENCODER_HIDDEN_STATES,
    p=DECODER_HIDDEN_STATES,
    y_dim=Y_DIM,
    dropout=DROPOUT
)

model = Model(
    darnn,
    'adam',
    'mse',
    device=device
)

save_to = 'checkpoint_torch.hdf5'

history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=7),
        ModelCheckpoint(
            str(save_to),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            keep_only_last_best=True
        )
    ]
)


def predict(model, X):
    with torch.no_grad():
        return model(X)


train_y_hat = predict(darnn, train_X.cuda())
plt.figure()
plt.plot(
    scale.inverse_transform(train_y_hat.cpu().repeat(1,raw_data.shape[1]).numpy())[:,-1],
    'k',
    scale.inverse_transform(train_y.repeat(1,raw_data.shape[1]).numpy())[:,-1],
    'r'
)
plt.show()

val_y_hat = predict(darnn, val_X.cuda())
plt.figure()
plt.plot(
    scale.inverse_transform(val_y_hat.cpu().repeat(1,raw_data.shape[1]).numpy())[:,-1],
    'k',
    scale.inverse_transform(val_y.repeat(1,raw_data.shape[1]).numpy())[:,-1],
    'r'
)
plt.show()


# Evaluation Metrics

S1 = pd.DataFrame(scale.inverse_transform(val_y_hat.cpu().repeat(1,raw_data.shape[1]).numpy())[:,-1])
S1['prev'] = S1[0].shift(1)
S1['ACC'] = S1[0] - S1['prev']
S1['ACC'] = S1['ACC'].mask(S1['ACC'] > 0, 1)
S1['ACC'] = S1['ACC'].mask(S1['ACC'] < 0, 0)
S1 = S1.dropna()

S2 = pd.DataFrame(scale.inverse_transform(val_y.repeat(1,raw_data.shape[1]).numpy())[:,-1])
S2['prev'] = S2[0].shift(1)
S2['ACC'] = S2[0] - S2['prev']
S2['ACC'] = S2['ACC'].mask(S2['ACC'] > 0, 1)
S2['ACC'] = S2['ACC'].mask(S2['ACC'] < 0, 0)
S2 = S2.dropna()

precision_score(S2['ACC'], S1['ACC'])
accuracy_score(S2['ACC'], S1['ACC'])
recall_score(S2['ACC'], S1['ACC'])


# Load Model checkpoint

# model.load_weights('checkpoint_torch.hdf5')
