import pandas
import json
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

CONSUMPTION_PATH = './real_consumption'
CONSUMPTION_KEY = 'hourlyConsumptions'
CONSUMPTION_FILE_LIST = []

PREDICTION_PATH = './next_day_pred'
PREDICTION_KEY= 'loadEstimationPlanList'
PREDICTION_FILE_LIST = []

# Klasör altındaki tüm dosyaları açıp, json dosyasını pandas dataframe'e çevirir.
# Her dosyada bulunan son 24 eleman dataframe içine eklenmez,
# bu veriler ortalama ve maximum gibi değerleri içermektedir.
def convert_to_df(_path, _key):
    _df = pandas.DataFrame()
    for file in os.listdir(_path):
        f = os.path.join(_path, file)
        if os.path.isfile(f):
            _json = json.dumps(
                json.load(open(f))[_key]
                , indent=2
            )
            df = pandas.read_json(_json)
            df.drop(df.tail(24).index, inplace=True)
            _df = _df.append(df, ignore_index=True)
    _df = _df.sort_values(['date'])
    _df = _df.reset_index(drop=True)
    return _df


# Tüketim ve tahmin verilerini dataframe'e çevirir.
# Daha sonrasında bu iki dataframe birleştirilir ve CSV olarak kaydedilir.
# Bu işlemlerin ardından tüm algoritmalar bu kaydedilen CSV dosyası üzerinden işlem yapar.
def get_main_df():
    consumption_df = convert_to_df(_path=CONSUMPTION_PATH, _key=CONSUMPTION_KEY)
    prediction_df = convert_to_df(_path=PREDICTION_PATH, _key=PREDICTION_KEY)

    main_df = pandas.merge(consumption_df, prediction_df, on='date')
    print(main_df)
    print("-"*40)
    print(main_df.info())
    main_df.to_csv('./all_data.csv', sep=',', index=False)
    return main_df

get_main_df()
