# %%
# N20230337 Ozan Koyuk
# VBM 695 Bitirme Projesi
# 11.2022
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from numpy.random import seed
from numpy import array
from tensorflow import random 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from tabulate import tabulate   # improves printing tables on console
from statistics import mean
from time import time
import os


def prepare_tabulate(real_cons, pred_cons, total_data, total_time, r2):
    tabulate_txt = []
    all_errors = []
    real_cons = real_cons.to_list()
    for i in range(24):
        yhat = pred_cons[i][0]
        expected = real_cons[i]

        # Calculate real and expected error
        error = round((abs(yhat - expected) / expected) * 100.0, 2)   
        tabulate_txt.append([yhat, expected, error])
        all_errors.append(error)
    
    # Prepare data for tabulate and txt file
    tabulate_txt.append(["-"*12, "-"*12, "-"*12])
    tabulate_txt.append(['Model',f"LSTM", ""])
    tabulate_txt.append(['Data Size', f"{total_data}", ""])
    tabulate_txt.append(['Average Error', round(mean(all_errors), 3), ""])
    tabulate_txt.append(['Time in sec.', f"{total_time}", ""])
    tabulate_txt.append(['R^2', "%.3f" % r2, ""])

    print(
        tabulate(tabulate_txt, headers=['Predicted(MWh)', 'Expected(MWh)', 'Error (%)'])
        )
    return tabulate_txt

# Plot graph with given data
def plotter(
    list_24_hours, 
    real_24_data, 
    label_real, 
    predicted_24_data, 
    label_predict, 
    xlabel, 
    ylabel, 
    title
    ):

    main_plot = plt.figure(1, figsize=(30, 30))
    plt.rcParams['font.size'] = '25'

    # Real data
    plt.plot(
        list_24_hours,
        real_24_data,
        color='black',
        linewidth=2,
        label=label_real
    )

    # Prediction
    plt.plot(
        list_24_hours,
        predicted_24_data,
        color='red',
        linewidth=2,
        label=label_predict
    )

    plt.gcf().autofmt_xdate()
    plt.xticks(list_24_hours)
    plt.legend(frameon=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()

    return main_plot


def data_split(sequence, TIMESTAMP):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + TIMESTAMP
        if end_ix > len(sequence)-1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# To give filenames, I use timestamp
_timestamp = int(dt.now().timestamp())

# If folder doesn't exist, then create it.
MYDIR = ("LSTM")
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)

# location,datetime,temp,dew_point,humidity,wind,wind_speed,wind_gust,pressure,condition
# 1 hour of data contains 2 30mins data.
# Import Google Trends Data
dataset = pd.read_csv("./all_data.csv",  names=['date', 'consumption', 'lep'], header=None, skiprows=1)
dataset.head()
dataset['consumption'] = dataset['consumption'].astype('float64').fillna(0.0)
dataset['lep'] = dataset['lep'].astype('float64')
print('SA')
print(dataset.info())
#dataset['date'] = pd.to_datetime(dataset['date'])

last_24_data = dataset.iloc[-24:] # remove last 24 hours to predict them
next_24_hours = [
    pd.to_datetime(last_24_data['date'].iloc[0]) + timedelta(hours=x) for x in range(0,24,1)
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

# Epoch -> one iteration over the entire dataset
N_EPOCHS = 8

# Batch_size -> divide dataset and pass into neural network.
BATCH_SIZE = 32

# Parse and divide data into size of 24 hour of data.
TIMESTAMP = 24

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
model = Sequential()
model.add(
    LSTM(
        120,
        activation='relu',
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
        )
    )
# Dropout: blocks random data for the given probability to next iteration.
model.add(Dropout(0.2))
model.add(LSTM(120, return_sequences=True))
model.add(Dropout(0.2))

# Final iteration needs no return_sequences because its the final step.
model.add(LSTM(120))

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

# End timer
stop = time()

# convert predicted values into real values
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train)
y_test_descaled = sc.inverse_transform(y_test)

y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]

main_plot = plotter(
    list_24_hours = _24_hours,
    real_24_data=dataset['consumption'].iloc[-24:],
    label_real='Real Time Consumption',
    predicted_24_data=y_predicted_descaled[-24:],
    label_predict='Predicted Consumption',
    xlabel='Time',
    ylabel='Consumption (MWh)',
    title=f"Comparison of Predicted-Real Time Electricity Consumption for the date :{predicted_date} with LSTM"
)
# Save graph into folder
main_plot.figure.savefig(f"./LSTM/LSTM_{N_EPOCHS}_Epochs_{predicted_date}_{_timestamp}.png")

r2 = r2_score(dataset['consumption'].iloc[-24:], y_predicted_descaled[-24:])

tabulate_txt = prepare_tabulate(
    real_cons=dataset['consumption'].iloc[-24:],
    pred_cons=y_predicted_descaled[-24:],
    total_data=len(dataset),
    total_time=round(stop-start, 3),
    r2=r2
    )

with open(f"./LSTM/LSTM_{N_EPOCHS}_Epochs_{predicted_date}_{_timestamp}.txt", 'w') as f:
    print(tabulate(tabulate_txt, headers=['Predicted(MWh)', 'Expected(MWh)', 'Error (%)']), file=f)
    

#                                                           
#   ___                   _  __                 _    
#  / _ \ ______ _ _ __   | |/ /___  _   _ _   _| | __
# | | | |_  / _` | '_ \  | ' // _ \| | | | | | | |/ /
# | |_| |/ / (_| | | | | | . \ (_) | |_| | |_| |   < 
#  \___//___\__,_|_| |_| |_|\_\___/ \__, |\__,_|_|\_\
#                                   |___/            
#  _   _   ____     ___    ____    _____    ___    _____   _____   _____ 
# | \ | | |___ \   / _ \  |___ \  |___ /   / _ \  |___ /  |___ /  |___  |
# |  \| |   __) | | | | |   __) |   |_ \  | | | |   |_ \    |_ \     / / 
# | |\  |  / __/  | |_| |  / __/   ___) | | |_| |  ___) |  ___) |   / /  
# |_| \_| |_____|  \___/  |_____| |____/   \___/  |____/  |____/   /_/                                           
#
#