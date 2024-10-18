# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:56:24 2024

@author: PC
"""



import requests
import time

# Alpha Vantage API key와 기본 URL
API_KEY = 'V2CQCYOFZUGLS5IS'
url = 'https://www.alphavantage.co/query'

# 환율 조회 함수
def get_exchange_rate(from_currency, to_currency):
    params = {
        'function': 'CURRENCY_EXCHANGE_RATE',
        'from_currency': from_currency,
        'to_currency': to_currency,
        'apikey': API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    try:
        exchange_rate = data['Realtime Currency Exchange Rate']['5. Exchange Rate']
        return float(exchange_rate)
    except KeyError:
        print("API 요청에 문제가 발생했습니다.")
        return None

# 분 단위로 환율 조회
def monitor_exchange_rate(from_currency, to_currency, interval=60):
    while True:
        rate = get_exchange_rate(from_currency, to_currency)
        if rate:
            print(f'{from_currency} to {to_currency} Exchange Rate: {rate}')
        time.sleep(interval)

# JPY to KRW 환율 1분마다 조회
monitor_exchange_rate('JPY', 'KRW', interval=60)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


cpath = r'C:\mscode\private\JPY_KRW 과거 데이터.csv'
df = pd.read_csv(cpath)


jong = np.array(df['종가'], dtype=float)
si = np.array(df['시가'], dtype=float)

jongsi = []
for i in range(len(jong)-1, -1, -1):
    jongsi.append(si[i])
    jongsi.append(jong[i])
jongsi = np.array(jongsi)    
plt.plot(jongsi)

start_date = df.iloc[-1,0]
end_date = df.iloc[0,0]

start_date = datetime.strptime(start_date.replace(" ", ""), '%Y-%m-%d')
end_date = datetime.strptime(end_date.replace(" ", ""), '%Y-%m-%d')

# 두 날짜 사이의 차이 계산
total_days = (end_date - start_date).days

#%%
ws = 10
thr = 1

fsave = []
for ws in [10, 20, 25, 30, 35, 40]:
    for thr in [0.3, 0.4, 0.5, 1, 2]:
        krw, qut = 1, 0
        zsave = np.zeros(len(jongsi)) * np.nan
        
        for t in range(ws, len(jongsi)):
            m = np.mean(jongsi[t-ws:t])
            s = np.std(jongsi[t-ws:t])
            zsave[t] = (jongsi[t] - m) / s
        
        buy = -1
        csave = []
        for t in range(ws, len(jongsi)):
             if zsave[t] < -thr and not(isbuy):
                 isbuy = True
                 amount = krw / jongsi[t]
                 qut = amount
                 krw = 0
        
             elif zsave[t] > thr and isbuy:
                 krw += qut * jongsi[t]
                 qut = 0
                 isbuy = False
        
             current_value = krw + (jongsi[t]*qut)
             csave.append(current_value)
             # print(t, current_value)

        fsave.append([ws, thr, csave[-1]])
        print(ws, thr, csave[-1])

fsave = np.array(fsave)
mix = np.argmax(fsave[:,-1])

print(fsave[mix])
ys = total_days/365.25
print('APR', (fsave[mix][-1] - 1) / ys)

#%%

ws, thr = int(fsave[mix][0]), fsave[mix][1]

krw, qut = 1, 0
zsave = np.zeros(len(jongsi)) * np.nan

for t in range(ws, len(jongsi)):
    m = np.mean(jongsi[t-ws:t])
    s = np.std(jongsi[t-ws:t])
    zsave[t] = (jongsi[t] - m) / s

buy = -1
csave = []
for t in range(ws, len(jongsi)):
     if zsave[t] < -thr and not(isbuy):
         isbuy = True
         amount = krw / jongsi[t]
         qut = amount
         krw = 0

     elif zsave[t] > thr and isbuy:
         krw += qut * jongsi[t]
         qut = 0
         isbuy = False

     current_value = krw + (jongsi[t]*qut)
     csave.append(current_value)
     # print(t, current_value)
plt.plot(csave)


























































