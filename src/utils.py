# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from apis import load_awos_by_point, load_ec_by_airport

NWP_DELAY_HOUR = {
    'EC': 6
}

NWP_START_HOUR = {
    'EC': 0
}

NWP_FREQ = {
    'EC': 12
}

NWP_FCST_LENGTH = {
    'EC': 36
}


def get_nwp_start_time(nwp_type, forecast_time: datetime):
    """查询给定时刻时能够获得的指定预报类型的最新一次预报的预报时间"""
    point_hour = forecast_time.hour - NWP_DELAY_HOUR[nwp_type]
    possible_hour = ((point_hour - NWP_START_HOUR[nwp_type]) // NWP_FREQ[nwp_type]) * NWP_FREQ[nwp_type] + \
                    NWP_START_HOUR[nwp_type]
    return forecast_time - timedelta(hours=point_hour + NWP_DELAY_HOUR[nwp_type] - possible_hour)


def load_nwp_data(start_time: datetime, end_time=None, days=None):
    """请求指定时间范围内的NWP预报，预报字段包括U10, V10, SPD10, DIR10, T2, TD2, PSFC"""
    nwp_type = 'EC'
    last_nwp_time = get_nwp_start_time(nwp_type, start_time)
    hour_span = (start_time - last_nwp_time).seconds // 3600 + (start_time - last_nwp_time).days * 24
    if end_time is not None:
        days = (end_time - start_time).days
    ec_df = load_ec_by_airport(last_nwp_time, days=days, start_point=hour_span)
    return ec_df


def load_nwp_data_mock(filename):
    with open(filename) as fid:
        data_dct = defaultdict(dict)
        for line in fid:
            fields = line.strip('\n').split('\t')
            if fields[1] == 'SLP':
                continue
            ec_time = datetime.strptime(fields[0], '%Y%m%d%H')
            forecast_time = (ec_time + timedelta(hours=12))
            for idx in range(-12, 24):
                data_dct[forecast_time][f'{fields[1]}.{idx}'] = float(fields[idx + 12 + 2])
    return [pd.DataFrame(data_dct).transpose()]


def pivot_arr_by_date(arr, tag):
    meta_df = arr.to_frame(name='obs')
    meta_df['date'] = arr.index.floor('d')
    meta_df['hour'] = arr.index.hour
    new_df = meta_df.pivot(columns='hour', index='date', values='obs')
    yesterday_df = new_df.copy()
    yesterday_df.index = yesterday_df.index + timedelta(days=1)
    yesterday_df.columns = [x - 24 for x in yesterday_df.columns]
    obs_mat = pd.concat([yesterday_df, new_df], axis=1)
    obs_mat.columns = [f'obs_{tag}.{x}' for x in obs_mat.columns]
    return obs_mat


def load_obs_mat(airport, point, start_time, days):
    obs_df = load_awos_by_point(airport, point, start_time, days=days)
    obs_ws_mat = pivot_arr_by_date(obs_df['obs_ws'], 'ws')
    obs_wd_mat = pivot_arr_by_date(obs_df['obs_wd'], 'wd')
    return pd.concat([obs_ws_mat, obs_wd_mat], axis=1)


"""
def load_obs_data_mock(filename):
    obs_data = pd.read_csv(filename, header=None, names=['time', 'obs'], sep='\t')
    obs_data['date'] = pd.to_datetime(obs_data['time'] // 10000, format='%Y%m%d')
    obs_data['hour'] = obs_data['time'] // 100 % 100
    obs_data = obs_data.pivot(columns='hour', index='date', values='obs')
    yesterday_obs = obs_data.copy()
    yesterday_obs.index = yesterday_obs.index + timedelta(days=1)
    yesterday_obs.columns = [x - 24 for x in yesterday_obs.columns]
    obs_mat = pd.concat([yesterday_obs, obs_data], axis=1)
    obs_mat.columns = [f'obs.{x}' for x in obs_mat.columns]
    return obs_mat
"""

if __name__ == '__main__':
    df_obs = load_obs_mat(airport='ZBAA', point='18L', start_time=datetime(2019, 4, 1), days=50)
    # df_ec = load_nwp_data(start_time=datetime(2019, 8, 20), days=6)
    pass
