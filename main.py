from algorithms import lstm, arima, fbprophet
from tabulate import tabulate   # improves printing tables on console

# LSTM
RUN_LSTM = True

# ARIMA
# If you want to find the best order for ARIMA, set FIND_BEST_ORDER to True.
RUN_ARIMA = True
FIND_BEST_ORDER = False

# PROPHET
RUN_PROPHET = True

# Comparison list
compare_list = ['Hours', 'PROPHET Error %', 'ARIMA Error %', 'LSTM Error %', 'EPİAŞ Error %']
general_comparison_tabulate = [
    *[[] for _ in range(27)],
]

# TODO: update this function with compare_list
def parse_tabulated_results(tabulated_results):
    hours = ["{:02d}:00".format(i) for i in range(24)]
    hours.append('------------')
    for index, row in enumerate(general_comparison_tabulate):
        if index <= 24:
            general_comparison_tabulate[index] = [
                hours[index],
                tabulated_results[0][index][-2],
                tabulated_results[1][index][-2],
                tabulated_results[2][index][-2],
                tabulated_results[2][index][-1] or tabulated_results[0][index][-1]
            ]
        elif index == 25:
            general_comparison_tabulate[index] = [
                'Average Errors',
                tabulated_results[0][index][-2],
                tabulated_results[1][index][-2],
                tabulated_results[2][index][-2],
                tabulated_results[2][index][-1] or tabulated_results[0][index][-1]
            ]
        elif index == 26:
            general_comparison_tabulate[index] = [
                'R^2',
                tabulated_results[0][index][-2],
                tabulated_results[1][index][-2],
                tabulated_results[2][index][-2],
                tabulated_results[2][index][-1] or tabulated_results[0][index][-1]
            ]


def main():
    lstm_results = [*[['', ''] for _ in range(27)]]
    arima_results = [*[['', ''] for _ in range(27)]]
    prophet_results = [*[['', ''] for _ in range(27)]]
    if RUN_LSTM:
        lstm_results = lstm.run_lstm()
    if RUN_ARIMA:
        arima_results = arima.run_arima(FIND_BEST_ORDER)
    if RUN_PROPHET:
        prophet_results = fbprophet.run_prophet()

    print('prophet_results', prophet_results)
    print('lstm_results', lstm_results)
    print('arima_results', arima_results)
    parse_tabulated_results(
        [prophet_results, arima_results, lstm_results]
    )

    tabulated_results = tabulate(general_comparison_tabulate, headers=compare_list)
    print(tabulated_results)


if __name__ == "__main__":
    main()
