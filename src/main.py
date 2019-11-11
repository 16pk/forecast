# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import argparse
import configparser
from sklearn.externals import joblib

from src.forecast_task import ForecastTask, load_task, save_task
from src.utils import load_nwp_data, load_obs_data
from src.utils import load_nwp_data_mock, load_obs_data_mock


# 训练模型
# 配置文件：
#   站点名称，经纬度
#   模型版本 模型存储路径
#   训练天数 预报开始时刻


def parse_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file, encoding='UTF-8')
    params = {}
    params['site_name'] = config['site_info'].get('site_name')
    params['latitude'] = config['site_info'].getfloat('latitude')
    params['longitude'] = config['site_info'].getfloat('longitude')
    params['training_days'] = config['model'].getint('training_days')
    params['start_hour'] = config['model'].getint('forecast_start_hour')
    params['nwp_sources'] = config['model'].get('nwp_source').split(',')
    return params


def load_data(fcst_start_time, n_days, config):
    nwp_data_list = [load_nwp_data(nwp, fcst_start_time, n_days, config['latitude'], config['longitude']) for nwp in
                     config['nwp_sources']]
    obs_data = load_obs_data(config['site_name'], fcst_start_time - timedelta(days=1), n_days + 1)
    return nwp_data_list, obs_data


def load_data_mock():
    nwp_data_list = load_nwp_data_mock('../data/ec_fcst_2018030112_2018103112.txt')
    obs_data = load_obs_data_mock('../data/obs_2018030112_2018103112_site_01.txt')
    return nwp_data_list, obs_data


def model_training(fcst_config):
    forecast_start_time = datetime.utcnow().replace(minute=0, hour=0, second=0, microsecond=0) \
                          - timedelta(days=fcst_config['training_days'] + 1) \
                          + timedelta(hours=fcst_config['start_hour'])
    # nwp_data_list, obs_data = load_data(forecast_start_time, fcst_config['training_days'], fcst_config)
    # todo mock
    nwp_data_list, obs_data = load_data_mock()
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
    fcst_config = parse_config('config_template')
    # model_training(fcst_config)
    task_obj = load_task('site_01', 0)
    pred, obs, ws_ec = prediction_mock(task_obj)
    print(pred)
    exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'predict'))
    parser.add_argument('--config')
    parser.add_argument('--site')
    parser.add_argument('--forecast_time')
    args = parser.parse_args()
    if args.stage == 'train':
        assert args.config is not None, 'Please input config file when training!'
        fcst_config = parse_config(args.config)
        model_training(fcst_config)
    elif args.stage == 'predict':
        assert args.site is not None, 'Please input site name when forecasting!'
        assert args.forecast_time is not None, 'Please input proper forecast time!'
        forecast_time = datetime.strptime(args.forecast_time)
        task_obj = load_task(args.site, forecast_time.hour)
        prediction(task_obj, forecast_time)

# python main.py train --config configFile
# python main.py predict --site siteName --forecast_time {YYYYMMDDHH}
