#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

import random

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame
from scipy import stats
from sklearn.model_selection import train_test_split

from data.based.file_types import FileTypes
from data.based.sampling_types import Sampling
from data.based.transformers_enums import TransformersType

seed = 2021
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)


class BasedDataset:

    def __init__(self, cfg, dataset_type):

        self._cfg = cfg
        self.dataset_type = dataset_type
        self.dataset_address = cfg.DATASET.DATASET_ADDRESS
        self.target_col = cfg.DATASET.TARGET
        self.dataset_description_file = cfg.DATASET.DATASET_BRIEF_DESCRIPTION

        if self.dataset_description_file is not None:
            self.about = self.__open_txt_file(self.dataset_description_file)

        self.load_dataset()
        self.df = self.df_main.copy()
        self.pca = None
        self.encoded_data = None
        self.scaled_data = None

    def load_dataset(self):
        """
        load dataset from csv file to dataframe
        """
        if self.dataset_type == FileTypes.CSV:
            self.df_main = self.__create_csv_dataframe()
        else:
            raise ValueError('dataset should be CSV file')

    def drop_cols(self):
        """
        drop columns from df

        """
        if self._cfg.DATASET.DROP_COLS is not None:
            cols = list(self._cfg.DATASET.DROP_COLS)
            self.df = self.df.drop(labels=cols, axis=1)

    def transformation(self, data: DataFrame, trans_type=None):
        """
        change the distribution of data by using log transformation, ...

        :param data:
        :param trans_type:
        :return:
        """

        try:
            if trans_type is None:
                cols = [*self._cfg.TRANSFORMATION]
                cols = [x.lower() for x in cols]
                cols = [x for x in cols if x in data.columns]
                _min = data[cols].min()
                for index, val in _min.iteritems():
                    if val <= 0:
                        data[index] = data[index] + 1 - val
                for col in cols:
                    if col in data.columns:
                        trans_type = self._cfg.TRANSFORMATION[col.upper()]
                        if trans_type == TransformersType.LOG:
                            data[col] = np.log(data[col])
                        elif trans_type == TransformersType.SQRT:
                            data[col] = np.sqrt(data[col])
                        elif trans_type == TransformersType.BOX_COX:
                            data[col] = stats.boxcox(data[col])[0]
            else:
                _min = data.min()
                for index, val in _min.iteritems():
                    if val <= 0:
                        data[index] = data[index] + 1 - val
                if trans_type == TransformersType.LOG:
                    data = np.log(data)
                elif trans_type == TransformersType.SQRT:
                    data = np.sqrt(data)
                elif trans_type == TransformersType.BOX_COX:
                    for col in data.columns:
                        data[col] = stats.boxcox(data[col])[0]


        except Exception as e:
            print('transform can not be applied ', e)

        return data

    def categorical_features(self, data=None):
        """
        select just categorical features from df

        :param data:
        :return:
        """
        if data is None:
            return self.df.select_dtypes(include=['object']).columns.tolist()
        else:
            return data.select_dtypes(include=['object']).columns.tolist()

    def numerical_features(self, data=None):
        """
         select just numerical features from df

        :param data:
        :return:
        """
        if data is None:
            return self.df.select_dtypes(exclude=['object']).columns.tolist()
        else:
            return data.select_dtypes(exclude=['object']).columns.tolist()

    def select_columns(self, data, cols=None, just_numerical=False):
        """
        select columns from df

        :param data:
        :param cols: array of columns that will be selected, None means select numerical features
        :param just_numerical:
        :return:
        """
        if cols is None or just_numerical:
            cols = self.numerical_features(data=data)
        return data[cols]

    def split_to(self, test_size=0.10, val_size=0.10, has_validation=False, use_pca=False, random_state=seed):
        """
        split dataset to train, test, validation set.

        :param test_size: size of test set
        :param val_size: size of validation set
        :param has_validation: set True if validation set is required
        :param use_pca: split dataset from pca components
        :param random_state:
        :return:
        """
        _X, _y = self.__samples_and_labels(use_pca=use_pca)

        _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=test_size, random_state=random_state)
        if has_validation:
            _X_train, _X_val, _y_train, y_val = train_test_split(_X_train, _y_train, test_size=val_size,
                                                                 random_state=random_state)
            return _X_train, _X_val, _X_test, _y_train, y_val, _y_test
        else:
            return _X_train, _X_test, _y_train, _y_test

    def generate_new_column_name(self, col, prefix):
        """
        generate new name for columns

        :param col:
        :param prefix:
        :return:
        """
        return '{}_{}'.format(col, prefix)

    def __samples_and_labels(self, use_pca=False):
        """
        return data as X and target values as y

        :param use_pca: select X from pca
        :return: data and label from df or pca
        """
        _X = None
        _y = None
        if use_pca:
            if self.pca is not None:
                _X = self.pca.copy().drop(labels=[self.target_col], axis=1)
                _y = self.pca[self.target_col].copy()
            else:
                print('pca data frame is not provided')
        else:
            _X = self.df.copy().drop(labels=[self.target_col], axis=1)
            _y = self.df[self.target_col].copy()

        return _X, _y

    def resampling(self, X, y):
        """
        resample dataset if you have imbalance data

        :param X: data
        :param y: targets
        :return: return new data with resampling strategy
        """

        if self._cfg.BASIC.SAMPLING_STRATEGY is None:
            raise ValueError(" SAMPLING_STRATEGY is None, Check the defaults.py")

        steps = []
        if type(self._cfg.BASIC.SAMPLING_STRATEGY) is tuple:
            sampling_types = [*self._cfg.BASIC.SAMPLING_STRATEGY]
            for smp in sampling_types:
                step = self.__resampling_pipeline(sampling_type=smp)
                steps.append(step)
        else:
            sampling_types = self._cfg.BASIC.SAMPLING_STRATEGY
            step = self.__resampling_pipeline(sampling_type=sampling_types)
            steps.append(step)

        pipeline = Pipeline(steps=steps)
        X, y = pipeline.fit_resample(X, y)
        return X, y

    def __resampling_pipeline(self, sampling_type):
        """
        create a pipeline for resampling data

        :param sampling_type:
        :return: a pipeline contains resampling strategies
        """
        steps = None

        if sampling_type == Sampling.RANDOM_UNDER_SAMPLING:

            params = {
                'sampling_strategy': self._cfg.RANDOM_UNDER_SAMPLER.SAMPLING_STRATEGY,
                'random_state': self._cfg.RANDOM_UNDER_SAMPLER.RANDOM_STATE,
                'replacement': self._cfg.RANDOM_UNDER_SAMPLER.REPLACEMENT
            }
            random_under_sampler = RandomUnderSampler(**params)
            steps = ('random_under_sampler', random_under_sampler)
        elif sampling_type == Sampling.RANDOM_OVER_SAMPLING:
            params = {
                'sampling_strategy': self._cfg.RANDOM_OVER_SAMPLER.SAMPLING_STRATEGY,
                'random_state': self._cfg.RANDOM_OVER_SAMPLER.RANDOM_STATE,
                # 'shrinkage': self._cfg.RANDOM_OVER_SAMPLER.SHRINKAGE
            }
            random_over_sampler = RandomOverSampler(**params)
            steps = ('random_over_sampler', random_over_sampler)
        elif sampling_type == Sampling.SMOTE:
            params = {
                'sampling_strategy': self._cfg.SMOTE.SAMPLING_STRATEGY,
                'random_state': self._cfg.SMOTE.RANDOM_STATE,
                'k_neighbors': self._cfg.SMOTE.K_NEIGHBORS,
                'n_jobs': self._cfg.SMOTE.N_JOBS
            }
            smote = SMOTE(**params)
            steps = ('smote', smote)
        elif sampling_type == Sampling.SMOTENC:
            params = {
                'categorical_features': self._cfg.SMOTENC.CATEGORICAL_FEATURES,
                'sampling_strategy': self._cfg.SMOTENC.SAMPLING_STRATEGY,
                'random_state': self._cfg.SMOTENC.RANDOM_STATE,
                'k_neighbors': self._cfg.SMOTENC.K_NEIGHBORS,
                'n_jobs': self._cfg.SMOTENC.N_JOBS
            }
            smotenc = SMOTENC(**params)
            steps = ('smotenc', smotenc)
        elif sampling_type == Sampling.SVMSMOTE:
            params = {
                'sampling_strategy': self._cfg.SVMSMOTE.SAMPLING_STRATEGY,
                'random_state': self._cfg.SVMSMOTE.RANDOM_STATE,
                'k_neighbors': self._cfg.SVMSMOTE.K_NEIGHBORS,
                'n_jobs': self._cfg.SVMSMOTE.N_JOBS,
                'm_neighbors': self._cfg.SVMSMOTE.M_NEIGHBORS,
                # 'svm_estimator': self._cfg.SMOTE.SVM_ESTIMATOR,
                'out_step': self._cfg.SVMSMOTE.OUT_STEP
            }
            svmsmote = SVMSMOTE(**params)
            steps = ('svmsmote', svmsmote)

        return steps

    def __create_csv_dataframe(self):
        """
        read data from csv file
        :return: pandas dataframe
        """
        return pd.read_csv(self.dataset_address, delimiter=';')

    def __open_txt_file(self, desc):
        """
        read contents from txt file
        :param desc:
        :return: contents as text
        """
        return open(desc, 'r').read()

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: DataFrame):
        self._df = df

    @property
    def pca(self):
        return self._pca

    @pca.setter
    def pca(self, value):
        self._pca = value

    @property
    def df_main(self):
        return self._df_main

    @df_main.setter
    def df_main(self, value):
        self._df_main = value

    @property
    def dataset_address(self):
        return self._dataset_address

    @dataset_address.setter
    def dataset_address(self, address):
        self._dataset_address = address

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        self._dataset_type = value

    @property
    def target_col(self):
        return self._target_col

    @target_col.setter
    def target_col(self, target):
        self._target_col = target

    @property
    def targets(self):
        return self.df_main[self.target_col]

    @property
    def about(self):
        return self._about

    @about.setter
    def about(self, about):
        self._about = about

    @property
    def dataset_description_file(self):
        return self._dataset_description_file

    @dataset_description_file.setter
    def dataset_description_file(self, value):
        self._dataset_description_file = value

    @property
    def encoded_data(self):
        return self._encoded_data

    @encoded_data.setter
    def encoded_data(self, value):
        self._encoded_data = value

    @property
    def scaled_data(self):
        return self._scaled_data

    @scaled_data.setter
    def scaled_data(self, value):
        self._scaled_data = value
