# %%
# N20230337 Ozan Koyuk
# VBM 695 Bitirme Projesi
# 11.2022
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from numpy.random import seed
from numpy import round as np_round
from tensorflow import random 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime as dt
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from tabulate import tabulate   # improves printing tables on console
from time import time
import os
from helpers import *

# To give filenames, I use timestamp
_timestamp = int(dt.now().timestamp())

# If folder doesn't exist, then create it.
if not LSTM_FOLDER_CHECK:
    os.makedirs(LSTM_FOLDER)

# Use the data prepared by crawler and converter codes.
dataset = pd.read_csv(
    CSV_FOLDER,
    names=HEADERS,
    header=None,
    skiprows=1
)
dataset.head()

# convert string data to float
dataset['consumption'] = dataset['consumption'].astype('float64')
dataset['lep'] = dataset['lep'].astype('float64')
print(dataset.info())

first_date = dt.strptime(dataset.iloc[0]['date'], '%Y-%m-%d %H:%M:%S+03:00').strftime('%d.%m.%Y')
end_date = dt.strptime(dataset.iloc[-25]['date'], '%Y-%m-%d %H:%M:%S+03:00').strftime('%d.%m.%Y')
next_24_hours = [
    dt.strptime(x, '%Y-%m-%d %H:%M:%S+03:00') 
    for x in dataset.iloc[-24:]['date'].to_list()
]

# Remove last 24 hours to predict them.
dataset = dataset.iloc[:-24:]

# Start timer
start = time()

# Perform medial filter
dataset['consumption'] = medfilt(dataset['consumption'], 3)

# Apply gaussian filter with sigma=1.2
dataset['consumption'] = gaussian_filter1d(dataset['consumption'], 1.2)

_24_hours = [
    dt.strftime(_hr, '%H:%M')
    for _hr in next_24_hours
]
predicted_date = dt.strftime(next_24_hours[0], '%d.%m.%Y')

seed(1)
random.set_seed(1)
# Number of days to train from. %80 of the data will be used to train.
train_days = int(len(dataset)*0.8)

# Number of days to be predicted. %20 of the data will be used to test.
testing_days = len(dataset) - train_days

train_set = dataset[0:train_days].reset_index(drop=True)
test_set = dataset[train_days: train_days+testing_days].reset_index(drop=True)

# Get consumption values into training and testing sets
training_set = train_set.iloc[:, 2:3].values
testing_set = test_set.iloc[:, 2:3].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(testing_set)

X_train, y_train = data_split(training_set_scaled, TIMESTAMP)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test, y_test = data_split(testing_set_scaled, TIMESTAMP)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Apply stacked LSTM wih 2 drop-outs
# activations: 'relu' / 'sigmoid' / 'tanh' / 'hardtanh' / 'leekly'
# return_sequences: True -> to pass results to the next iteration of LSTM
# input_shape: (X_train.shape[1], 1) -> shape is TIMESTAMP value
# total hidden layer count = 24
model = Sequential()
model.add(
    LSTM(
        24,
        activation='relu',
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
        )
    )
# Dropout: blocks random data for the given probability to next iteration.
model.add(Dropout(0.2))
model.add(LSTM(24, return_sequences=True))
model.add(Dropout(0.2))

# Final iteration needs no return_sequences because its the final step.
model.add(LSTM(24))

# When return_sequences is set to False,
# Dense is applied to the last time step only.
model.add(Dense(1))

# Most used optimizers: adam, sgd, adadelta, adamax, nadam
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train,
    y_train,
    epochs=N_EPOCHS,
    batch_size=BATCH_SIZE
)

# loss -> array of loss values for every iteration
# epochs -> count of epochs
loss = history.history['loss']
epochs = range(len(loss))

y_predicted = model.predict(X_test)

# convert predicted values into real values
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train)
y_test_descaled = sc.inverse_transform(y_test)

y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]

main_plot = plotter(
    list_24_hours = _24_hours,
    real_24_data=dataset['consumption'].iloc[-24:],
    real_24_predicted_data=dataset['lep'].iloc[-24:],
    label_real='Real Time Consumption',
    predicted_24_data=y_predicted_descaled[-24:],
    label_predict='LSTM Prediction',
    xlabel='Time',
    ylabel='Consumption (MWh)',
    title=f"Comparison of Predicted-Real Time Electricity Consumption for the date :{predicted_date} with LSTM",
    label_real_predict='EPİAŞ Prediction'
)

# Save graph into folder
main_plot.figure.savefig(f"{LSTM_FOLDER}/LSTM_{N_EPOCHS}_Epochs_{predicted_date}_{_timestamp}.png")

r2_real = r2_score(dataset['consumption'].iloc[-24:], dataset['lep'].iloc[-24:])
r2_lstm = r2_score(dataset['consumption'].iloc[-24:], y_predicted_descaled[-24:])

# End timer
stop = time()

headers=[
    'Expected(MWh)', 
    'Predicted LSTM(MWh)', 
    'Predicted EPİAŞ(MWh)', 
    'Error LSTM(%)', 
    'Error EPİAŞ(%)'
]

tabulate_txt = prepare_tabulate(
    first_date=first_date,
    end_date=end_date,
    predicted_date=predicted_date,
    real_cons=dataset['consumption'].iloc[-24:],
    real_pred_list=dataset['lep'].iloc[-24:],
    new_pred_cons=np_round(y_predicted_descaled[-24:], 2),
    total_time=round(stop-start, 3),
    r2_real=r2_real,
    r2_lstm=r2_lstm,
    headers=headers,
    N_EPOCHS=N_EPOCHS,
    BATCH_SIZE=BATCH_SIZE,
    TIMESTAMP=TIMESTAMP
    )

with open(f"{LSTM_FOLDER}/LSTM_{N_EPOCHS}_Epochs_{predicted_date}_{_timestamp}.txt", 'w') as f:
    tabulated_results = tabulate(tabulate_txt, headers=headers)
    print(tabulated_results)
    print(tabulated_results, file=f)


#   ___ _____   _    _   _   _  _______   ___   _ _  __
#  / _ \__  /  / \  | \ | | | |/ / _ \ \ / / | | | |/ /
# | | | |/ /  / _ \ |  \| | | ' / | | \ V /| | | | ' / 
# | |_| / /_ / ___ \| |\  | | . \ |_| || | | |_| | . \ 
#  \___/____/_/   \_\_| \_| |_|\_\___/ |_|  \___/|_|\_\
#  _   _ ____   ___ ____  _____  ___ _______________ 
# | \ | |___ \ / _ \___ \|___ / / _ \___ /___ /___  |
# |  \| | __) | | | |__) | |_ \| | | ||_ \ |_ \  / / 
# | |\  |/ __/| |_| / __/ ___) | |_| |__) |__) |/ /  
# |_| \_|_____|\___/_____|____/ \___/____/____//_/   
