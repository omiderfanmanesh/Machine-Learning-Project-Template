#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler

from data.based.scale_types import ScaleTypes


class Scalers:
    def __init__(self, cfg):
        self._cfg = cfg

    def __get_scaler(self, scale_type):
        """
        get scaler object
        :param scale_type:
        :return: scaler object, name of scaler
        """
        scl = None
        scl_name = None
        if scale_type == ScaleTypes.MIN_MAX:
            scl = MinMaxScaler()
            scl_name = 'min_max_scaler'
        elif scale_type == ScaleTypes.STANDARD:
            scl = StandardScaler()
            scl_name = 'min_max_scaler'
        elif scale_type == ScaleTypes.MAX_ABS:
            scl = MaxAbsScaler()
            scl_name = 'max_abs_scaler'
        elif scale_type == ScaleTypes.ROBUST:
            scl = RobustScaler()
            scl_name = 'robust_scaler'

        return scl, scl_name

    def do_scale(self, data=None, X_train=None, X_test=None):
        """
        get data and apply scaler function

        :param data: change scale of whole data
        :param X_train: change scale of train set and fit scaler to it
        :param X_test: transform test data scale based on train set
        :return: scaled dataframe
        """
        if data is None:
            _train_scale, _test_scale = self.__get_scaled_values(X_train=X_train, X_test=X_test)
            _train_df = pd.DataFrame(data=_train_scale, columns=X_train.columns)
            _test_df = pd.DataFrame(data=_test_scale, columns=X_train.columns)
            return _train_df, _test_df
        else:
            _data_scale = self.__get_scaled_values(data=data)
            _data_df = pd.DataFrame(data=_data_scale, columns=data.columns)
            return _data_df

    def __get_scaled_values(self, data=None, X_train=None, X_test=None):
        """
        get scaler configs and objects and apply scaler
        :param data:
        :param X_train:
        :param X_test:
        :return:
        """
        scl_type = self._cfg.SCALER
        scl, scl_name = self.__get_scaler(scale_type=scl_type)
        if data is None:
            return self.__apply(scl=scl, X_train=X_train, X_test=X_test)
        else:
            return self.__apply(scl=scl, data=data)

    def __apply(self, scl, data=None, X_train=None, X_test=None):
        """
        apply scaler to data
        :param scl: scaler obj
        :param data: data
        :param X_train: train set
        :param X_test: test set
        :return: scaled data
        """
        if data is None:
            scl.fit(X_train)
            train_scale = scl.transform(X_train)
            test_scale = None
            if X_test is not None:
                test_scale = scl.transform(X_test)
            return train_scale, test_scale
        else:
            scl.fit(data)
            train_scale = scl.transform(data)
            return train_scale
