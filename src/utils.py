# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from apis import load_awos_by_point, load_ec_by_airport, load_wrf_by_airport

NWP_DELAY_HOUR = {
    'EC': 6,
    'WRFVAR': 6
}

NWP_START_HOUR = {
    'EC': 0,
    'WRFVAR': 12
}

NWP_FREQ = {
    'EC': 12,
    'WRFVAR': 24
}

NWP_FCST_LENGTH = {
    'EC': 36,
    'WRFVAR': 36
}


def get_nwp_start_time(nwp_type, forecast_time: datetime):
    """查询给定时刻时能够获得的指定预报类型的最新一次预报的预报时间"""
    point_hour = forecast_time.hour - NWP_DELAY_HOUR[nwp_type]
    possible_hour = ((point_hour - NWP_START_HOUR[nwp_type]) // NWP_FREQ[nwp_type]) * NWP_FREQ[nwp_type] + \
                    NWP_START_HOUR[nwp_type]
    return forecast_time - timedelta(hours=point_hour + NWP_DELAY_HOUR[nwp_type] - possible_hour)


def load_nwp_data(nwp_type, point, start_time: datetime, end_time=None, days=None):
    """请求指定时间范围内的NWP预报，预报字段包括U10, V10, SPD10, DIR10, T2, TD2, PSFC"""
    last_nwp_time = get_nwp_start_time(nwp_type, start_time)
    hour_span = (start_time - last_nwp_time).seconds // 3600 + (start_time - last_nwp_time).days * 24
    if end_time is not None:
        days = (end_time - start_time).days
    if nwp_type == 'EC':
        nwp_df = load_ec_by_airport(site_name=point, start_time=last_nwp_time, days=days, start_point=hour_span)
    elif nwp_type == 'WRFVAR':
        nwp_df = load_wrf_by_airport(site_name=point, start_time=last_nwp_time, days=days, start_point=hour_span)
    return nwp_df


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


if __name__ == '__main__':
    df_obs = load_obs_mat(airport='ZBAA', point='18L', start_time=datetime(2019, 4, 3), days=1)
    df_wrf = load_nwp_data('WRFVAR', point='18L', start_time=datetime(2019, 4, 20), days=6)
    pass
