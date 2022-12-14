# %%
# N20230337 Ozan Koyuk
# VBM 695 Bitirme Projesi
# 11.2022
from time import time
from datetime import datetime as dt
from prophet import Prophet
from tabulate import tabulate   # improves printing tables on console
from pandas import to_datetime
from sklearn.metrics import r2_score
import warnings
from .helpers import *

# Disable warnings for Jupyter etc.
warnings.filterwarnings("ignore")


def run_prophet():
    # If folder doesn't exist, then create it.
    if not PROPHET_FOLDER_CHECK:
        os.makedirs(PROPHET_FOLDER)

    # To give filenames, I use timestamp
    _timestamp = int(dt.now().timestamp())

    df = pd.read_csv(
            CSV_FOLDER,
            names=HEADERS,
            header=None,
            skiprows=1  # pass header
        )
    # remove timezone from the date due to prophet rules.
    df.info()

    # remove last 24 hours to predict them
    last_24_data = df.iloc[-24:]

    # find start and end date, prepare next 24 hours
    first_date = dt.strptime(df.iloc[0]['date'], '%Y-%m-%d %H:%M:%S+03:00').strftime('%d.%m.%Y')
    end_date = dt.strptime(df.iloc[-25]['date'], '%Y-%m-%d %H:%M:%S+03:00').strftime('%d.%m.%Y')

    df['date'] = pd.to_datetime(df.date).dt.tz_localize(None)

    next_24_hours = [
            dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')
            for x in df.iloc[-24:]['date'].to_list()
        ]
    _24_hours = [
            dt.strftime(_hr, '%H:%M')
            for _hr in next_24_hours
        ]

    predicted_date = dt.strftime(next_24_hours[0], '%d.%m.%Y')

    # Convert predicted date into dataframe
    next_24_hours = pd.DataFrame(next_24_hours)
    next_24_hours.columns = ['ds']
    next_24_hours['ds'] = to_datetime(next_24_hours['ds'])

    # Rename column names for Prophet Algorithm
    data = df.rename(columns={'consumption': 'y', 'date': 'ds'})

    start = time()

    # growth: linear, logistic, flat
    model = Prophet(growth='linear', yearly_seasonality=True)

    if USE_ONLY_ONE_YEAR:
        model_fit = model.fit(data[-365*24:])
        # 365 * 24 hours = 1 year
    else:
        model_fit = model.fit(data)

    forecast = model.predict(next_24_hours)
    end = time()

    y_pred = [round(_yhat, 2) for _yhat in forecast['yhat'].values]

    # Prepare X-axis for graph -> ["00:00", "01:00",...]
    only_24_hours = [hrs.strftime('%H:%M') for hrs in next_24_hours['ds']]

    # Use plotter to plot graph
    # TODO: fix multiple line problem
    main_plot = plotter(
            list_24_hours=_24_hours,
            real_24_data=df['consumption'].iloc[-24:],
            real_24_predicted_data=df['lep'].iloc[-24:],
            label_real='Real Time Consumption',
            predicted_24_data=y_pred,
            label_predict='Prophet Prediction',
            xlabel='Time',
            ylabel='Consumption (MWh)',
            title=f"Comparison of Predicted-Real Time Electricity Consumption for:{predicted_date} with Prophet Algorithm",
            label_real_predict='EPİAŞ Prediction'
        )

    # Save it to use in reports etc.
    main_plot.figure.savefig(f"{PROPHET_FOLDER}/PROPHET_{predicted_date}_{_timestamp}.png")

    headers = [
        'Expected(MWh)',
        'Predicted PROPHET(MWh)',
        'Predicted EPİAŞ(MWh)',
        'Error PROPHET(%)',
        'Error EPİAŞ(%)'
    ]

    r2_real = r2_score(df['consumption'], df['lep'])
    r2_prophet = r2_score(df['consumption'].iloc[-24:], y_pred)

    tabulate_txt = prepare_tabulate(
        first_date=first_date,
        end_date=end_date,
        predicted_date=predicted_date,
        real_cons=df['consumption'].iloc[-24:],
        real_pred_list=df['lep'].iloc[-24:],
        new_pred_cons=np_round(y_pred, 2),
        total_time=round(end - start, 3),
        r2_real=r2_real,
        r2_score=r2_prophet,
        headers=headers,
        N_EPOCHS=N_EPOCHS,
        BATCH_SIZE=BATCH_SIZE,
        TIMESTAMP=TIMESTAMP,
        model_name='PROPHET',
    )

    with open(f"{PROPHET_FOLDER}/PROPHET_{predicted_date}_{_timestamp}.txt", 'w') as f:
        tabulated_results = tabulate(tabulate_txt, headers=headers)
        # print(tabulated_results)
        print(tabulated_results, file=f)
    print(f"PROPHET calculation completed in total time: {round(end - start, 3)} seconds")
    print("#" * 50)
    return tabulate_txt

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

# %%
