# -*- coding: utf-8 -*-
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from retry import retry

HOST = 'http://161.189.11.216:8090'


def load_awos_by_point(airport, site_name, start_time, end_time=None, days=None):
    if end_time is None:
        end_time = start_time + timedelta(days=days)
    params = {'datacode': 'uu',
              'airport': airport,
              'runwayName': site_name,
              'dataset': 'WS10X,WD10X',
              'starttime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
              'endtime': end_time.strftime('%Y-%m-%d %H:%M:%S')}
    url = f'{HOST}/gis/BJPEK/RunwaysHistoryByRunwayName'
    data_json = get_json(url, params)
    ws = _str2num(data_json['runwayName'][0][site_name]['WS10X'][::60])
    wd = _str2num(data_json['runwayName'][0][site_name]['WD10X'][::60])
    time_idx = pd.date_range(start_time, end_time, freq=timedelta(hours=1))
    return pd.DataFrame({'obs_ws': ws, 'obs_wd': wd}, index=time_idx).iloc[:-1].copy()


def load_ec_by_airport(start_time, end_time=None, days=None, start_point=0):
    if end_time is None:
        end_time = start_time + timedelta(days=days)
    meteo_list = ['U10', 'V10', 'SPD10', 'DIR10', 'T2', 'TD2', 'PSFC']
    params = {'starttime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
              'endtime': end_time.strftime('%Y-%m-%d %H:%M:%S'),
              'dataSetList': ','.join(meteo_list)}
    url = f'{HOST}/gis/BJPEK/ECData'
    data_json = get_json(url, params)['data'][0]
    for ec_datestr in data_json:
        meta_dct = {}
        for x in data_json[ec_datestr]:
            meta_dct.update(x)
        data_json[ec_datestr] = meta_dct
    df_lst = []
    for meteo in meteo_list:
        meta_df = pd.DataFrame({x: data_json[x][meteo] for x in data_json}, dtype=float).transpose()
        meta_df.columns = [f'{meteo}.{x-start_point}' for x in meta_df.columns]
        df_lst.append(meta_df)
    final_df = pd.concat(df_lst, axis=1)
    final_df.index = pd.to_datetime(final_df.index, format='%Y%m%d%H%M%S') + timedelta(hours=start_point)
    return final_df.iloc[:-1].copy()


def _str2num(list_obj):
    return [float(x) if x not in ('null', ' ') else np.nan for x in list_obj]


@retry(tries=5, delay=10, backoff=2)
def get_json(url, params):
    res = requests.get(url, params)
    res.raise_for_status()
    return res.json()


if __name__ == '__main__':
    df = load_awos_by_point(airport='ZBAA', site_name='18L', start_time=datetime(2019, 9, 20), days=3)
    ec_df = load_ec_by_airport(datetime(2019, 9, 20), days=3)
    print(df.shape)
    print(df.head())
