from algorithms import lstm, arima
from tabulate import tabulate   # improves printing tables on console

# LSTM
RUN_LSTM = False

# ARIMA
# If you want to find the best order for ARIMA, set FIND_BEST_ORDER to True.
RUN_ARIMA = True
FIND_BEST_ORDER = False

# Comparison list
compare_list = ['Hours', 'ARIMA Error %', 'LSTM Error %', 'EPİAŞ Error %']
general_comparison_tabulate = [
    *[[] for _ in range(27)],
]


def parse_tabulated_results(tabulated_results):
    hours = ["{:02d}:00".format(i) for i in range(24)]
    hours.append('------------')
    for index, row in enumerate(general_comparison_tabulate):
        if index <= 24:
            general_comparison_tabulate[index] = [
                hours[index],
                tabulated_results[0][index][-2],
                tabulated_results[1][index][-2],
                tabulated_results[1][index][-1] or tabulated_results[0][index][-1]
            ]
        elif index == 25:
            general_comparison_tabulate[index] = [
                'Average Errors',
                tabulated_results[0][index][-2],
                tabulated_results[1][index][-2],
                tabulated_results[1][index][-1] or tabulated_results[0][index][-1]
            ]
        elif index == 26:
            general_comparison_tabulate[index] = [
                'R^2',
                tabulated_results[0][index][-2],
                tabulated_results[1][index][-2],
                tabulated_results[1][index][-1] or tabulated_results[0][index][-1]
            ]


def main():
    lstm_results = [*[['', ''] for _ in range(27)]]
    arima_results = [*[['', ''] for _ in range(27)]]
    if RUN_LSTM:
        lstm_results = lstm.run_lstm()
    if RUN_ARIMA:
        arima_results = arima.run_arima(FIND_BEST_ORDER)

    parse_tabulated_results(
        [arima_results, lstm_results]
    )

    tabulated_results = tabulate(general_comparison_tabulate, headers=compare_list)
    print(tabulated_results)


if __name__ == "__main__":
    main()
