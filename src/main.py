# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import argparse
import configparser
from sklearn.externals import joblib

from src.forecast_task import ForecastTask, load_task, save_task
from src.utils import load_nwp_data, load_obs_mat
from src.utils import load_nwp_data_mock, load_obs_data_mock
from src.apis import load_awos_by_point, load_ec_by_airport


# 训练模型
# 配置文件：
#   站点名称，经纬度
#   模型版本 模型存储路径
#   训练天数 预报开始时刻


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


def load_data(start_time, n_days, config):
    nwp_data_list = [load_nwp_data(start_time, days=n_days) for nwp in config['nwp_sources']]
    obs_data = load_obs_mat(config['airport'], config['site_name'], start_time, n_days)
    return nwp_data_list, obs_data


def load_data_mock():
    nwp_data_list = load_nwp_data_mock('../data/ec_fcst_2018030112_2018103112.txt')
    obs_data = load_obs_data_mock('../data/obs_2018030112_2018103112_site_01.txt')
    return nwp_data_list, obs_data


def model_training(train_start_time, train_days, fcst_config):
    nwp_data_list, obs_data = load_data(train_start_time, train_days, fcst_config)
    # todo mock
    # nwp_data_list, obs_data = load_data_mock()
    task_obj = ForecastTask(fcst_config)
    task_obj.train(nwp_data_list, obs_data)
    save_task(task_obj)


def prediction(task_obj, forecast_time):
    fcst_config = task_obj.get_config()
    nwp_data_list, obs_data = load_data(forecast_time, 1, fcst_config)
    fcst_result = task_obj.predict(nwp_data_list, obs_data)
    return fcst_result


def prediction_mock(task_obj):
    fcst_config = task_obj.get_config()
    # nwp_data_list, obs_data = load_data(forecast_time, 1, fcst_config)
    nwp_data_list, obs_data = load_data_mock()
    nwp_data_list[0] = nwp_data_list[0].loc[nwp_data_list[0].index >= datetime(2018, 10, 10)]
    obs_data = obs_data.loc[obs_data.index >= datetime(2018, 10, 10)]
    fcst_result = task_obj.predict(nwp_data_list, obs_data)
    return fcst_result, obs_data, nwp_data_list[0][[f'EC.ws.{x}' for x in range(24)]]


if __name__ == '__main__':
    # fcst_config = parse_config('config_template')
    # # model_training(fcst_config)
    # task_obj = load_task('site_01', 0)
    # pred, obs, ws_ec = prediction_mock(task_obj)
    # print(pred)
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'predict'))
    parser.add_argument('--airport')
    parser.add_argument('--site')
    parser.add_argument('--train_start_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--train_end_time', help='format: "%Y%m%d%H%M%S')
    parser.add_argument('--train_days', type=int, default=0)
    # parser.add_argument('--forecast_time')
    args = parser.parse_args()
    assert args.airport is not None and args.site is not None, 'Airport and site name are needed!'
    if args.stage == 'train':
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
                       # 'train_start_time': train_start_time,
                       # 'training_days': train_days,
                       'start_hour': train_start_time.hour,
                       'nwp_sources': ['EC']}
        # fcst_config = parse_config(args.config)
        model_training(train_start_time, train_days, fcst_config)
    elif args.stage == 'predict':
        # assert args.site is not None, 'Please input site name when forecasting!'
        # assert args.forecast_time is not None, 'Please input proper forecast time!'
        forecast_time = datetime.strptime(args.forecast_time, format='%Y%m%d%H%M%S')
        task_obj = load_task(args.site, forecast_time.hour)
        prediction(task_obj, forecast_time)

# python main.py train --config configFile
# python main.py predict --site siteName --forecast_time {YYYYMMDDHH}
