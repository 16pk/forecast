# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb


def save_task(task_obj):
    model_file_name = f'model_{task_obj.site_name}_{task_obj.start_hour}.joblib'
    joblib.dump(task_obj, model_file_name)


def load_task(site_name, forecast_start_hour):
    model_file_name = f'model_{site_name}_{forecast_start_hour}.joblib'
    task_obj = joblib.load(model_file_name)
    return task_obj


class ForecastTask(object):
    """
    配置站点信息及预报参数，调度预报模型
    """

    def __init__(self, params):
        """
        site_name, latitude, longitude, training_days, start_hour, nwp_sources
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
            # 风速，湿球温度与气温差值，预报风速误差
            for idx in range(-12, 24):
                datamat[f'{nwp}.ws.{idx}'] = np.sqrt(
                    datamat[f'{nwp}.U10.{idx}'] ** 2 + datamat[f'{nwp}.V10.{idx}'] ** 2)
                datamat[f'{nwp}.rh_delta.{idx}'] = datamat[f'{nwp}.T.{idx}'] - datamat[f'{nwp}.RH.{idx}']
                datamat[f'{nwp}.bias.{idx}'] = datamat[f'obs.{idx}'] - datamat[f'{nwp}.ws.{idx}']
            # 气压变，温度变，风速变
            for idx in range(0, 24):
                for span in (1, 3, 6, 12):
                    datamat[f'{nwp}.PSFC_{span}d.{idx}'] = datamat[f'{nwp}.PSFC.{idx}'] - datamat[
                        f'{nwp}.PSFC.{idx-span}']
                    datamat[f'{nwp}.T_{span}d.{idx}'] = datamat[f'{nwp}.T.{idx}'] - datamat[f'{nwp}.T.{idx-span}']
                    datamat[f'{nwp}.ws_{span}d.{idx}'] = datamat[f'{nwp}.ws.{idx}'] - datamat[f'{nwp}.ws.{idx-span}']
        return datamat

    def _get_feature_list(self, fc_hr):
        feat_list = [f'{self.nwp_sources[0]}.bias.{x}' for x in range(-12, 0)]
        for nwp in self.nwp_sources:
            feat_list += [f'{nwp}.U10.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.V10.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.ws.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.rh_delta.{x}' for x in range(-12, 24)] \
                         + [f'{nwp}.PSFC_{span}d.{fc_hr}' for span in (1, 3, 6, 12)] \
                         + [f'{nwp}.T_{span}d.{fc_hr}' for span in (1, 3, 6, 12)] \
                         + [f'{nwp}.ws_{span}d.{fc_hr}' for span in (1, 3, 6, 12)]
        return feat_list

    def train(self, nwp_data_list, obs, n_val_day=20):
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

        datamat = self.data_prepross(nwp_data_list, obs)
        for fc_hr in range(0, 24):
            print(f'Hour: {fc_hr}')
            feat_list = self._get_feature_list(fc_hr)
            y_tag = f'{self.nwp_sources[0]}.bias.{fc_hr}'
            x_train, y_train, x_val, y_val = _train_val_split(datamat, n_val_day, feat_list, y_tag)
            clf = xgb.XGBRegressor(
                booster='dart',
                learning_rate=0.03,
                n_estimators=300,
                subsample=0.3,
                colsample_bytree=0.35,
                max_depth=3,
                seed=42)
            clf.fit(x_train, y_train,
                    eval_set=[(x_val, y_val)],
                    eval_metric='rmse', verbose=0,
                    early_stopping_rounds=40)
            best_iter = clf.best_iteration
            print(f'Best iteration is: {best_iter}')
            clf = xgb.XGBRegressor(
                booster='dart',
                learning_rate=0.037,
                n_estimators=best_iter,
                subsample=0.3,
                colsample_bytree=0.35,
                max_depth=3,
                seed=42)
            clf.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
            self.models[fc_hr] = clf

    def predict(self, nwp_data_list, obs):
        datamat = self.data_prepross(nwp_data_list, obs)
        pred_dct = {}
        for fc_hr in range(0, 24):
            feat_list = self._get_feature_list(fc_hr)
            yhat = self.models[fc_hr].predict(datamat[feat_list])
            yhat = pd.Series(yhat, index=datamat.index)
            pred_dct[f'pred_{fc_hr}'] = yhat + datamat[f'{self.nwp_sources[0]}.ws.{fc_hr}']
        return pd.DataFrame(pred_dct)[[f'pred_{x}' for x in range(0, 24)]]


if __name__ == '__main__':
    pass
