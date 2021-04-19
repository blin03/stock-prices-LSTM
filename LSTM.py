import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




# Import training set
dataset_train = pd.read_csv('GOOG.csv')

cols = list(dataset_train)[1:6]




# Pull dates from dataset
datelist_train = list(dataset_train['Date'])
datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]

print('Training Dataset Shape == {}'.format(dataset_train.shape))
print('All Timestamps == {}'.format(len(datelist_train)))
print('Features: {}'.format(cols))




# Convert to matrix
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)

training_set = dataset_train.to_numpy()

print('Shape of training set == {}.'.format(training_set.shape))




# Scale dataset to standard values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])




# Create data structure
x_train = []
y_train = []

n_future = 60  # Number of days into the future to forecast
n_past = 90  # Number of previous days used to forecast

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    x_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future - 1: i + n_future, 0])

x_train, y_train = np.array(x_train), np.array(y_train)  # Convert list to array

print('x_train shape == {}'.format(x_train.shape))
print('y_train shape == {}'.format(y_train.shape))




# Create model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam

model = Sequential()

model.add(LSTM(units=64, return_sequences=True,
                input_shape=(n_past, dataset_train.shape[1]-1),
                bias_regularizer = tf.keras.regularizers.l2(0.015),
                kernel_regularizer = tf.keras.regularizers.l2(0.015),
                recurrent_regularizer = tf.keras.regularizers.l2(0.015))
          )
model.add(LSTM(units=10,
                return_sequences=False,
                bias_regularizer=tf.keras.regularizers.l2(0.015),
                kernel_regularizer=tf.keras.regularizers.l2(0.015),
                recurrent_regularizer=tf.keras.regularizers.l2(0.015))
          )
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.03), loss='mean_squared_error')

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=3, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')

history = model.fit(x_train, y_train,
                    shuffle=True,
                    epochs=30,
                    callbacks=[es, rlr, mcp, tb],
                    validation_split=0.2,
                    verbose=1,
                    batch_size=128,
                    )


datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

predictions_future = model.predict(x_train[-n_future:])

predictions_train = model.predict(x_train[n_past:])

def datetime_to_timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open Price']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open Price']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))


PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

print(PREDICTION_TRAIN.head(20))