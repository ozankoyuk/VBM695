import datetime
import requests
from dateutil.relativedelta import relativedelta
import json
import os

# şehirler :  api/consumption/şehir rest servisi
# piyasa takas fiyatı: api/market/piyasa tkas fiyat rest servisi
# GİP ağırlıklı ortalama fiyat: api/market/gip ağırlıklı ortalama fiyat rest servisi
# GİP özet: api/market/gün içi piyasası özet servisi
# bir sonraki gün tahmin: https://seffaflik.epias.com.tr/transparency/tuketim/tahmin/yuk-tahmin-plani.xhtml

# bir sonraki gün tahmin: /consumption/load-estimation-plan?startDate=2022-10-26T00:00:00&endDate=2022-10-27T00:00:00
# gerçekleşen tüketim: /consumption/real-time-consumption?startDate=2020-01-01T00:00:00Z&endDate=2022-10-26T00:00:00

baseURL = 'https://api.epias.com.tr/epias/exchange/transparency'
consumptionURL = '/consumption/real-time-consumption'
predictionURL = '/consumption/load-estimation-plan'

def check_folders():
    if not os.path.isdir('./next_day_pred'):
        os.mkdir(os.getcwd() + '/next_day_pred')
    if not os.path.isdir('./real_consumption'):
        os.mkdir(os.getcwd() + '/real_consumption')


def download_real_consumption():
    start_date, end_date, now = reset_dates()
    while start_date < end_date:
        now +=  relativedelta(months=+1)
        print(f'between: {start_date} / {now}')
        response_data = requests.get(f"{baseURL}{consumptionURL}?startDate={start_date.strftime('%Y-%m-%dT%H:%M:%S')}&endDate={now.strftime('%Y-%m-%dT%H:%M:%S')}")
        json_object = json.dumps(response_data.json()['body'], indent=2)
        
        with open(f"./real_consumption/{start_date.year}_{start_date.month}.json", "w") as outfile:
            outfile.write(json_object)
        start_date = start_date + relativedelta(months=+1)

def download_predictions():
    start_date, end_date, now = reset_dates()
    while now < end_date:
        now +=  relativedelta(months=+1)
        print(f'between: {start_date} / {now}')
        response_data = requests.get(f"{baseURL}{predictionURL}?startDate={start_date.strftime('%Y-%m-%dT%H:%M:%S')}&endDate={now.strftime('%Y-%m-%dT%H:%M:%S')}")
        json_object = json.dumps(response_data.json()['body'], indent=2)
        
        with open(f"./next_day_pred/{start_date.year}_{start_date.month}.json", "w") as outfile:
            outfile.write(json_object)
        start_date = start_date + relativedelta(months=+1)

def reset_dates():
    start_date = datetime.datetime(2018,1,1,0,0,0)
    end_date = datetime.datetime(2022,10,1,0,0,0)
    now = start_date
    return (start_date, end_date, now)

def crawl_data():
    check_folders()
    download_predictions()
    download_real_consumption()
