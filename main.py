from algorithms import lstm, arima

# LSTM
RUN_LSTM = False

# ARIMA
# If you want to find the best order for ARIMA, set FIND_BEST_ORDER to True.
RUN_ARIMA = True
FIND_BEST_ORDER = False

def main():
    if RUN_LSTM:
        lstm.run_lstm()
    if RUN_ARIMA:
        arima.run_arima(FIND_BEST_ORDER)


if __name__ == "__main__":
    main()
