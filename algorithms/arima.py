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
from datetime import datetime as dt

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
    print('Generate model with given order..')
    daily_prediction, result_txt, all_errors = generate_model(
        _data=history, 
        _model=model,
        _order=order,
        _last_24_data=real_consumption
        )
    end = time.time()

    # Use plotter to plot graph
    main_plot = plotter(
        list_24_hours=only_24_hours,  # 24 hours ["00:00", "01:00", ...]
        real_24_data=real_consumption['consumption'],   # real consumption
        label_real='Real Time Consumption',     # set pred line name
        predicted_24_data=daily_prediction,     # set pred line values
        label_predict='Predicted Consumption',  # set cons line name
        xlabel="Time",                          # set x axis
        ylabel="Consumption (MWh)",             # set y axis
        title=f"Comparison of Predicted-Real Time Electricity Consumption for the date :{predicted_date} with ARIMA"
        )

    # Save graph into folder
    print('Saving figure..')
    main_plot.figure.savefig(f"{ARIMA_FOLDER}/{model.__name__}_{order}_{predicted_date}_{_timestamp}.png")
    r2 = r2_score(real_consumption['consumption'], daily_prediction)

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

    print('Completed ARIMA..')
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
        expected = _last_24_data['consumption'].iloc[t]
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

# If folder doesn't exist, then create it.
if not ARIMA_FOLDER_CHECK:
    os.makedirs(ARIMA_FOLDER)

# Import Google Trends Data
df = pd.read_csv(
    CSV_FOLDER,
    names=HEADERS,
    header=None,
    skiprows=1
)
df.head()

first_date = dt.strptime(df.iloc[0]['date'], '%Y-%m-%d %H:%M:%S+03:00').strftime('%d.%m.%Y')
end_date = dt.strptime(df.iloc[-25]['date'], '%Y-%m-%d %H:%M:%S+03:00').strftime('%d.%m.%Y')
next_24_hours = [
    dt.strptime(x, '%Y-%m-%d %H:%M:%S+03:00') 
    for x in df.iloc[-24:]['date'].to_list()
]
real_consumption = df.iloc[-24:]

# Set X-axis data
only_24_hours = [hrs.strftime('%H:%M') for hrs in next_24_hours]

# Set predicted date for title and filename.
predicted_date = datetime.strftime(next_24_hours[0], '%d.%m.%Y')

if USE_ONLY_ONE_YEAR:
    data = df['consumption'].iloc[-365*24:]
    # 365 * 24 hours = 1 year
else:
    data = df['consumption']

history = [hourly_consumption for hourly_consumption in data]
model = ARIMA
print("Predicting..")
_order, _mean = predict_and_write_files(history=history, model=model, order=(1, 1, 1), only_24_hours=only_24_hours)

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
