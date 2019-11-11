# -*- coding: utf-8 -*-
"""
@author: liuhongjian
@contact: liuhongjian@duxiaoman.com
@created time: 2019-11-10
"""
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

NWP_DELAY_HOUR = {
    'EC': 4
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
    return forecast_time - timedelta(hours=point_hour - possible_hour)


def load_nwp_data(nwp_type, start_time: datetime, days, latitude, longitude):
    # todo: 需要基于接口内容确定
    # 需要保留预报时刻前12小时的气象预报，用于计算delta
    last_nwp_time = get_nwp_start_time(nwp_type, start_time)
    pass


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


def load_obs_data(site_name, start_time, days):
    pass


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


if __name__ == '__main__':
    pass
