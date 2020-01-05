# -*- coding: utf-8 -*-
"""
Usage:

使用制定时段的数据训练模型
    python main.py train
        --airport ZBAA
        --site 18L
        --train_start_time 20180901000000
        --train_end_time 20181015000000
        --nwp_sources EC
        --prefix ec_
使用当前时间前若干天的数据训练模型
    python main.py train
        --airport ZBAA
        --site 18L
        --train_days 400
        --nwp_sources EC
        --prefix ec_
"""
from datetime import datetime, timedelta
import argparse
import json

from forecast_task import ForecastTask, load_task, save_task
from utils import load_nwp_data, load_obs_mat


def load_data(start_time, n_days, config):
    nwp_data_list = [load_nwp_data(nwp, config['site_name'], start_time, days=n_days) for nwp in config['nwp_sources']]
    obs_data = load_obs_mat(config['airport'], config['site_name'], start_time - timedelta(days=1), n_days)
    obs_data = obs_data.loc[nwp_data_list[0].index]
    return nwp_data_list, obs_data


def model_training(train_start_time, train_days, fcst_config, prefix='./'):
    nwp_data_list, obs_data = load_data(train_start_time, train_days, fcst_config)
    task_obj = ForecastTask(fcst_config)
    task_obj.train(nwp_data_list, obs_data)
    save_task(task_obj, prefix)


def prediction(task_obj, forecast_start_time, forecast_days=1):
    fcst_config = task_obj.get_config()
    nwp_data_list, obs_data = load_data(forecast_start_time, forecast_days, fcst_config)
    fcst_result_arr = task_obj.predict(nwp_data_list, obs_data)

    nwp_str = '_'.join(task_obj.nwp_sources)
    start_time_str = forecast_start_time.strftime('%Y%m%d%H%M')
    end_time_str = (forecast_start_time + timedelta(days=forecast_days) - timedelta(hours=1)).strftime('%Y%m%d%H%M')
    result_file = f'{nwp_str}_AI_{task_obj.airport}_{task_obj.site_name}_0010m_SPD_{start_time_str}_{end_time_str}_3000M_NWP02.dat'
    with open(result_file, 'w') as fid:
        for x in fcst_result_arr:
            fid.write(f'{x:.3f}\n')
    return fcst_result_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'predict'))
    parser.add_argument('--airport')
    parser.add_argument('--site')
    parser.add_argument('--train_start_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--train_end_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--train_days', type=int, default=0)
    parser.add_argument('--nwp_sources', default='EC')
    parser.add_argument('--forecast_start_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--forecast_days', type=int, default=1)
    parser.add_argument('--prefix', default='./')
    args = parser.parse_args()
    assert args.airport is not None and args.site is not None, 'Airport and site name are needed!'
    if args.stage == 'train':
        nwp_sources = args.nwp_sources.split(',')
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
                       'nwp_sources': nwp_sources}
        model_training(train_start_time, train_days, fcst_config, args.prefix)
    elif args.stage == 'predict':
        forecast_time = datetime.strptime(args.forecast_start_time, '%Y%m%d%H%M%S')
        forecast_days = args.forecast_days
        task_obj = load_task(args.airport, args.site, forecast_time.hour, args.prefix)
        prediction(task_obj, forecast_time, forecast_days)
