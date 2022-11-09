from numpy import array, round as np_round
import matplotlib.pyplot as plt
from statistics import mean
import os
import pandas as pd

# MAIN
CSV_FOLDER = os.getcwd() + "/all_data.csv"
HEADERS = pd.read_csv(CSV_FOLDER).columns.to_list()
# date, consumption, lep

# ARIMA
USE_ONLY_ONE_YEAR = True
ARIMA_FOLDER = (os.getcwd() + "/ARIMA")
ARIMA_FOLDER_CHECK = os.path.isdir(ARIMA_FOLDER)

# LSTM
LSTM_FOLDER = (os.getcwd() + "/LSTM_RESULTS")
LSTM_FOLDER_CHECK = os.path.isdir(LSTM_FOLDER)
N_EPOCHS = 1  # Epoch -> one iteration over the entire dataset
BATCH_SIZE = 24  # Batch_size -> divide dataset and pass into neural network.
TIMESTAMP = 24  # Parse and divide data into size of 24 hour of data.


def prepare_tabulate(**kwargs):
    tabulate_txt = []
    real_errors = []
    lstm_errors = []
    real_cons = kwargs["real_cons"].to_list()
    real_pred_list = kwargs["real_pred_list"].to_list()

    for i in range(24):
        new_pred = np_round(kwargs["new_pred_cons"][i][0], 2)
        real_pred = round(real_pred_list[i], 2)
        expected = round(real_cons[i], 2)

        # Calculate real and expected error
        error_lstm = round((abs(new_pred - expected) / expected) * 100.0, 2)   
        error_real = round((abs(real_pred - expected) / expected) * 100.0, 2)   
        tabulate_txt.append([expected, round(new_pred, 2), real_pred, error_lstm, error_real])
        lstm_errors.append(error_lstm)
        real_errors.append(error_real)

    # Prepare data for tabulate and txt file
    tabulate_txt.append(["-"*12, "-"*12, "-"*12, "-"*12, "-"*12])
    tabulate_txt.append(['Average Errors','','', round(mean(lstm_errors), 3), round(mean(real_errors),3)])
    tabulate_txt.append(['R^2','','', "%.3f" % kwargs["r2_lstm"], "%.3f" % kwargs["r2_real"]])
    tabulate_txt.append(["-"*12, "-"*12, "-"*12, "-"*12, "-"*12])
    tabulate_txt.append(['Time in sec.', kwargs["total_time"], ""])
    tabulate_txt.append(['Start Date', kwargs["first_date"], '', '', ''])
    tabulate_txt.append(['End Date', kwargs["end_date"], '', '', ''])
    tabulate_txt.append(['Predicting', kwargs["predicted_date"], '', '', ''])
    tabulate_txt.append(["-"*12, "-"*12, "-"*12, "-"*12, "-"*12])
    tabulate_txt.append(['Epoch Count', kwargs["N_EPOCHS"], '', '', ''])
    tabulate_txt.append(['Batch Size', kwargs["BATCH_SIZE"], '', '', ''])
    tabulate_txt.append(['Timestamp', kwargs["TIMESTAMP"], '', '', ''])
    tabulate_txt.append(["-"*12, "-"*12, "-"*12, "-"*12, "-"*12])

    return tabulate_txt

# Plot graph with given data
def plotter(
    list_24_hours, 
    real_24_data,
    real_24_predicted_data,
    label_real, 
    predicted_24_data,
    label_predict, 
    xlabel, 
    ylabel, 
    title,
    label_real_predict
    ):

    main_plot = plt.figure(1, figsize=(30, 30))
    plt.rcParams['font.size'] = '25'

    # Real data
    plt.plot(
        list_24_hours,
        real_24_data,
        color='black',
        linewidth=3,
        label=label_real
    )

    # Prediction
    plt.plot(
        list_24_hours,
        predicted_24_data,
        color='red',
        linewidth=3,
        label=label_predict
    )

    # Real Predicted Data
    plt.plot(
        list_24_hours,
        real_24_predicted_data,
        color='green',
        linewidth=3,
        label=label_real_predict,
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
