# -*- coding: utf-8 -*-
"""
Usage:

使用制定时段的数据训练模型
    python main.py train --airport ZBAA --site 18L --train_start_time 20180901000000 --train_end_time 20181015000000
使用当前时间前若干天的数据训练模型
    python main.py train --airport ZBAA --site 18L --train_days 400
"""
from datetime import datetime, timedelta
import argparse
import json

from forecast_task import ForecastTask, load_task, save_task
from utils import load_nwp_data, load_obs_mat

"""
def parse_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file, encoding='UTF-8')
    params = {}
    params['airport'] = config['site_info'].get('airport')
    params['site_name'] = config['site_info'].get('site_name')
    params['training_days'] = config['model'].getint('training_days')
    params['start_hour'] = config['model'].getint('forecast_start_hour')
    params['nwp_sources'] = config['model'].get('nwp_source').split(',')
    return params
"""


def load_data(start_time, n_days, config):
    nwp_data_list = [load_nwp_data(start_time, days=n_days) for nwp in config['nwp_sources']]
    obs_data = load_obs_mat(config['airport'], config['site_name'], start_time, n_days)
    return nwp_data_list, obs_data


"""
def load_data_mock():
    nwp_data_list = load_nwp_data_mock('../data/ec_fcst_2018030112_2018103112.txt')
    obs_data = load_obs_data_mock('../data/obs_2018030112_2018103112_site_01.txt')
    return nwp_data_list, obs_data
"""


def model_training(train_start_time, train_days, fcst_config):
    nwp_data_list, obs_data = load_data(train_start_time, train_days, fcst_config)
    task_obj = ForecastTask(fcst_config)
    task_obj.train(nwp_data_list, obs_data)
    save_task(task_obj)


def prediction(task_obj, forecast_start_time, forecast_days=1):
    fcst_config = task_obj.get_config()
    nwp_data_list, obs_data = load_data(forecast_start_time, forecast_days, fcst_config)
    fcst_result_arr = task_obj.predict(nwp_data_list, obs_data)
    fcst_result_str = json.dumps(list(fcst_result_arr))
    result_file = f'ws_forecast_{task_obj.airport}_{task_obj.site_name}_{forecast_start_time.strftime("%Y%m%d")}_{forecast_days}.json'
    with open(result_file, 'w') as fid:
        fid.write(fcst_result_str)
    return fcst_result_arr


"""
def prediction_mock(task_obj):
    fcst_config = task_obj.get_config()
    # nwp_data_list, obs_data = load_data(forecast_time, 1, fcst_config)
    nwp_data_list, obs_data = load_data_mock()
    nwp_data_list[0] = nwp_data_list[0].loc[nwp_data_list[0].index >= datetime(2018, 10, 10)]
    obs_data = obs_data.loc[obs_data.index >= datetime(2018, 10, 10)]
    fcst_result = task_obj.predict(nwp_data_list, obs_data)
    return fcst_result, obs_data, nwp_data_list[0][[f'EC.ws.{x}' for x in range(24)]]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'predict'))
    parser.add_argument('--airport')
    parser.add_argument('--site')
    parser.add_argument('--train_start_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--train_end_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--train_days', type=int, default=0)
    # todo
    # parser.add_argument('--nwp_sources')
    parser.add_argument('--forecast_start_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--forecast_days', type=int, default=1)
    args = parser.parse_args()
    assert args.airport is not None and args.site is not None, 'Airport and site name are needed!'
    if args.stage == 'train':
        # todo
        # nwp_sources = args.nwp_sources.split(',')
        if args.train_days == 0:
            train_start_time = datetime.strptime(args.train_start_time, '%Y%m%d%H%M%S')
            train_end_time = datetime.strptime(args.train_end_time, '%Y%m%d%H%M%S')
            train_days = (train_end_time - train_start_time).days
        else:
            train_days = args.train_days
            train_start_time = datetime.utcnow().replace(minute=0, hour=0, second=0, microsecond=0) \
                               - timedelta(days=train_days + 1)
        fcst_config = {'airport': args.airport,
                       'site_name': args.site,
                       'start_hour': train_start_time.hour,
                       'nwp_sources': ['EC']}
        model_training(train_start_time, train_days, fcst_config)
    elif args.stage == 'predict':
        forecast_time = datetime.strptime(args.forecast_start_time, '%Y%m%d%H%M%S')
        forecast_days = args.forecast_days
        task_obj = load_task(args.airport, args.site, forecast_time.hour)
        prediction(task_obj, forecast_time, forecast_days)
