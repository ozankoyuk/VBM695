# %%
# N20230337 Ozan Koyuk
# VBM 695 Bitirme Projesi
# 11.2022
import pandas as pd
from datetime import timedelta, datetime
from statsmodels.tsa.arima.model import ARIMA
from tabulate import tabulate   # improves printing tables on console
from statistics import mean
import time
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from helpers import *

import warnings
warnings.filterwarnings('ignore')


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

def predict_and_write_files(*, history, model, order, only_24_hours):
    start = time.time()
    # Generate prediction
    daily_prediction, result_txt, all_errors = generate_model(
        _data=history, 
        _model=model,
        _order=order,
        _last_24_data=last_24_data
        )
    end = time.time()

    # Use plotter to plot graph
    main_plot = plotter(
        list_24_hours=only_24_hours,  # 24 hours ["00:00", "01:00", ...]
        real_24_data=last_24_data['Tuketim'],   # real consumption
        label_real='Real Time Consumption',     # set pred line name
        predicted_24_data=daily_prediction,     # set pred line values
        label_predict='Predicted Consumption',  # set cons line name
        xlabel="Time",                          # set x axis
        ylabel="Consumption (MWh)",             # set y axis
        title=f"Comparison of Predicted-Real Time Electricity Consumption for the date :{predicted_date} with ARIMA"
        )

    # Save graph into folder
    main_plot.figure.savefig(f"./ARIMA/{model.__name__}_{order}_{predicted_date}_{_timestamp}.png")
    r2 = r2_score(last_24_data['Tuketim'], daily_prediction)

    # Prepare data for tabulate and txt file
    result_txt.append(["-"*12, "-"*12, "-"*12])
    result_txt.append(['Model',f"ARIMA", f"{order}"])
    result_txt.append(['Data Size', f"{len(history)}", ""])
    result_txt.append(['Average Error', round(mean(all_errors), 3), ""])
    result_txt.append(['Time in sec.', "%.3f" % round(end-start, 2), ""])
    result_txt.append(['R^2', "%.3f" % r2, ""])

    print(
        tabulate(result_txt, headers=['Predicted(MWh)', 'Expected(MWh)', 'Error (%)'])
        )

    with open(f"./ARIMA/ARIMA_{order}_{predicted_date}_{_timestamp}.txt", 'w') as f:
        print(tabulate(result_txt, headers=['Predicted(MWh)', 'Expected(MWh)', 'Error (%)']), file=f)
    
    return order, mean(all_errors)


def generate_model(*, _data, _model, _order, _last_24_data):
    _daily_prediction = []
    _print_result = []
    _all_errors = []
    for t in range(len(_last_24_data)):
        # Create model
        model = _model(_data, order=_order)
        model_fit = model.fit()

        # Forecast the consumption
        yhat = round(model_fit.forecast()[0], 2)

        # Append list to use it later
        _daily_prediction.append(yhat)
        expected = _last_24_data['Tuketim'].iloc[t]
        _data.append(expected)

        # Calculate real and expected error
        error = round((abs(yhat - expected) / expected) * 100.0, 2)

        # Append list to use it later
        _all_errors.append(error)

        # Append for tabulate
        _print_result.append([yhat, expected, error])
    return _daily_prediction, _print_result, _all_errors


# To give filenames, I use timestamp
_timestamp = int(datetime.now().timestamp())

# Set data size
USE_ONLY_ONE_YEAR = True

# If folder doesn't exist, then create it.
MYDIR = (os.getcwd() + "/ARIMA")
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)

# Import Google Trends Data
df = pd.read_csv("./01012016-19102021.csv",  names=['Tarih', 'Saat', 'Tuketim'])
df.head()
# Convert str "37.430,65" -> float "37430.65" 
df['Tuketim'] = df['Tuketim'].str.replace('.','').astype(str)
df['Tuketim'] = df['Tuketim'].str.replace(',','.').astype(float)

# Merge 'Tarih' and 'Saat' to create a datetime object
# and remove 'Saat' column
# "01.01.2016","01:00" -> "01.01.2016 01:00"
df['Tarih'] = df['Tarih'] +" "+ df['Saat']
df['Tarih'] = pd.to_datetime(df['Tarih'])
df = df.drop(columns='Saat')

last_24_data = df.iloc[-24:] # remove last 24 hours to predict them
next_24_hours = [
    pd.to_datetime(last_24_data['Tarih'].iloc[0] + timedelta(hours=x),format='%Y%m%d%H%M') for x in range(0,24,1)
]

# Set X-axis data
only_24_hours = [hrs.strftime('%H:%M') for hrs in next_24_hours]

# Set predicted date for title and filename.
predicted_date = datetime.strftime(next_24_hours[0], '%d.%m.%Y')

if USE_ONLY_ONE_YEAR:
    data = df['Tuketim'].iloc[-365*24:]
    # 365 * 24 hours = 1 year
else:
    data = df['Tuketim']

history = [hourly_consumption for hourly_consumption in data]
model = ARIMA
_order, _mean = predict_and_write_files(history=history, model=model, order=(10, 1, 1), only_24_hours=only_24_hours)

#####################################################
#   After this line, calculating and comparing      #
#   all models part starts. This will calculate     #
#   every possible model (p,d,q) and saves the      #
#   result graph and data into current directory    #
#   This process takes about 7 hours due to         #
#   data size.                                      #
#---------------------------------------------------#
#   I've already found the best model with          #
#   this code and it is (10,1,1) which is used      #
#   to write the report.                            #
#####################################################

# p_values = [0, 1, 2, 4, 6, 8, 10]
# d_values = range(0, 3)
# q_values = range(0, 3)
# """
# p is the number of autoregressive terms,
# d is the number of nonseasonal differences needed for stationarity, and
# q is the number of lagged forecast errors in the prediction equation. 
# """
# best_order = ""
# best_error_mean = 100
# for p in p_values:
#     for d in d_values:
#         for q in q_values:
#             order = (p, d, q)
#             print(f"Starting new order: {order}")
#             _order, _mean = predict_and_write_files(history=history, model=model, order=order)
#             if _mean < best_error_mean:
#                 best_error_mean = _mean
#                 best_order = _order
# print(f"Best order -> {best_order} with mean error -> {best_error_mean}")

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
