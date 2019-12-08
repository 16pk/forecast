# -*- coding: utf-8 -*-
"""
预报任务的实现，任务对象的序列化与反序列化
"""
from sklearn.externals import joblib
import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb
from model_opt import get_best_params_by_bo


def save_task(task_obj, prefix='./'):
    model_file_name = f'{prefix}model_{task_obj.airport}_{task_obj.site_name}_{task_obj.start_hour}.joblib'
    joblib.dump(task_obj, model_file_name)


def load_task(airport, site_name, forecast_start_hour, prefix='./'):
    model_file_name = f'{prefix}model_{airport}_{site_name}_{forecast_start_hour}.joblib'
    task_obj = joblib.load(model_file_name)
    return task_obj


class ForecastTask(object):
    """
    配置站点信息及预报参数，调度预报模型
    """

    def __init__(self, params):
        """
        airport, site_name, start_hour, nwp_sources
        """
        self.__dict__ = params
        self.models = {}

    def get_config(self):
        config = self.__dict__.copy()
        config.pop('models')
        return config

    def data_prepross(self, nwp_data_list, obs):
        new_data_list = [obs]
        for nwp, nwp_data in zip(self.nwp_sources, nwp_data_list):
            new_columns = [f'{nwp}.{x}' for x in nwp_data.columns]
            nwp_data.columns = new_columns
            new_data_list.append(nwp_data)
        datamat = pd.concat(new_data_list, axis=1)
        for nwp in self.nwp_sources:
            # 已有风速、风向、UV分量、地表气压、地表温度、湿球温度
            # 湿球温度与气温差值，预报风速误差
            for idx in range(-12, 24):
                datamat[f'{nwp}.rh_delta.{idx}'] = datamat[f'{nwp}.T2.{idx}'] - datamat[f'{nwp}.TD2.{idx}']
                datamat[f'{nwp}.bias.{idx}'] = datamat[f'obs_ws.{idx}'] - datamat[f'{nwp}.SPD10.{idx}']
            # 气压变，温度变，风速变
            for idx in range(0, 24):
                for span in (1, 3, 6, 12):
                    datamat[f'{nwp}.PSFC_{span}d.{idx}'] = datamat[f'{nwp}.PSFC.{idx}'] - datamat[
                        f'{nwp}.PSFC.{idx-span}']
                    datamat[f'{nwp}.T2_{span}d.{idx}'] = datamat[f'{nwp}.T2.{idx}'] - datamat[f'{nwp}.T2.{idx-span}']
                    datamat[f'{nwp}.SPD10_{span}d.{idx}'] = datamat[f'{nwp}.SPD10.{idx}'] - datamat[
                        f'{nwp}.SPD10.{idx-span}']
        return datamat

    def _get_feature_list(self, fc_hr):
        feat_list = [f'{self.nwp_sources[0]}.bias.{x}' for x in range(-12, 0)]
        for nwp in self.nwp_sources:
            feat_list += [f'{nwp}.U10.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.V10.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.SPD10.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.DIR10.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.rh_delta.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.PSFC_{span}d.{fc_hr}' for span in (1, 3, 6, 12)] \
                         + [f'{nwp}.T2_{span}d.{fc_hr}' for span in (1, 3, 6, 12)] \
                         + [f'{nwp}.SPD10_{span}d.{fc_hr}' for span in (1, 3, 6, 12)]
        return feat_list

    def train(self, nwp_data_list, obs_data, n_val_day=20):
        def _train_val_split(datamat, n_val_day, feat_list, y_tag):
            latest_day = datamat.index.max()
            is_train = datamat.index <= latest_day - timedelta(days=n_val_day)
            is_val = datamat.index > latest_day - timedelta(days=n_val_day)
            x_train = datamat.loc[is_train, feat_list].copy()
            y_train = datamat.loc[is_train, y_tag].copy()
            x_val = datamat.loc[is_val, feat_list].copy()
            y_val = datamat.loc[is_val, y_tag].copy()
            valid_train = x_train.notnull().any(axis=1) & y_train.notnull()
            if not valid_train.all():
                x_train = x_train.loc[valid_train]
                y_train = y_train.loc[valid_train]
            valid_val = x_val.notnull().any(axis=1) & y_val.notnull()
            if not valid_val.all():
                x_val = x_val.loc[valid_val]
                y_val = y_val.loc[valid_val]
            return x_train, y_train, x_val, y_val

        datamat = self.data_prepross(nwp_data_list, obs_data)
        for fc_hr in range(0, 24):
            print(f'Hour: {fc_hr}')
            feat_list = self._get_feature_list(fc_hr)
            y_tag = f'{self.nwp_sources[0]}.bias.{fc_hr}'
            x_train, y_train, x_val, y_val = _train_val_split(datamat, n_val_day, feat_list, y_tag)
            best_params = get_best_params_by_bo(x_train, y_train, x_val, y_val)
            clf = xgb.XGBRegressor(booster='gbtree', n_estimators=300, verbosity=0, n_jobs=16, seed=42,
                                   reg_alpha=0.1, reg_lambda=0.1, **best_params)
            clf.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
            self.models[fc_hr] = clf

    def predict(self, nwp_data_list, obs_data):
        datamat = self.data_prepross(nwp_data_list, obs_data)
        pred_list = []
        for fc_hr in range(0, 24):
            feat_list = self._get_feature_list(fc_hr)
            yhat = self.models[fc_hr].predict(datamat[feat_list])
            yhat = pd.Series(yhat, index=datamat.index)
            pred_one = yhat + datamat[f'{self.nwp_sources[0]}.SPD10.{fc_hr}']
            pred_one.index = pred_one.index + timedelta(hours=fc_hr)
            pred_list.append(pred_one)
        return pd.concat(pred_list).sort_index()


if __name__ == '__main__':
    pass
