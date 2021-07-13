#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.


import copy
from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from data.based.transformers_enums import TransformersType


class BasedAnalyzer:
    def __init__(self, dataset, cfg):
        self._dataset = dataset
        self._df = dataset.df

    def head(self):
        self.df.head()

    def description(self, col=None):
        if self.dataset.dataset_description_file is not None and col is None:
            print("--------------- about dataset  -----------------")
            print(self.dataset.about)
            print('\n')
            print("--------------- description.txt ----------------")
            pprint(self.info())
            print('\n')
            print("--------------- description.txt ----------------")
            pprint(self.describe_dataframe())
            print('\n')

        if col is None:
            print("--------------- nan Values -----------------")
            print(self.missing_values().head(20))
            print('\n')
        else:
            print("--------------- nan Values of {} -----------------".format(col))
            print(self.missing_values(col=col))
            print('\n')

        if col is None:
            print("--------------- duplicates -----------------")
        else:
            print("--------------- duplicates of {} -----------------".format(col))

        print('Total number of duplicates: ', self.duplicates(col))
        print('\n')

        if col is None:
            print("------ Numerical/Categorical Features ------")
            print('Numerical Features: {}'.format(self.dataset.numerical_features()))
            print('number of Numerical Features: {}'.format(self.dataset.numerical_features().__len__()))
            print('Categorical Features: {}'.format(self.dataset.categorical_features()))
            print('number of Categorical Features: {}'.format(self.dataset.categorical_features().__len__()))
            print('\n')

        if col is None:
            print("--------------- skew & kurt -----------------")
        else:
            print("--------------- skew & kurt of {} -----------------".format(col))

        print('calculate skewness and kurtosis of numerical features')
        print(self.skew_kurt(col=col))
        # print(
        #     '\n* skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable '
        #     'about its mean. \nnegative skew commonly indicates that the tail is on the left side of the distribution, '
        #     'and positive skew indicates that the tail is on the right.\n ')
        # print('* kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random '
        #       'variable. Like skewness,\n kurtosis describes the shape of a probability distribution and there are '
        #       'different ways of quantifying it for a theoretical distribution \nand corresponding ways of estimating '
        #       'it from a sample from a population.')
        print('\n')

        if col is None:
            print("----------------- quantiles -----------------")
        else:
            print("--------------- quantiles of {} -----------------".format(col))

        print(self.quantiles(col=col))
        print('\n')

        if col is None:
            print("----------------- is target balanced? -----------------")
            print(self.count_by(col=self.target_col))
            print('\n')
        else:
            print("----------------- Top 15 values in column of {} -----------------".format(col))
            print(self.count_by(col=col).head(15))
            print('\n')

    def count_by(self, col):
        new_df = self.df[col].value_counts().sort_values(ascending=False).reset_index()
        new_df.columns = ['value', 'counts']
        return new_df

    def missing_values(self, col=None):
        if col is None:
            total = self.df.isnull().sum().sort_values(ascending=False)
            percent = (self.df.isnull().sum() / self.df.isnull().count()).sort_values(ascending=False)
            missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        else:
            total = self.df[col].isnull().sum()
            percent = (self.df[col].isnull().sum() / self.df[col].isnull().count())

            missing_data = {'total': total, 'percentage': percent}
        return missing_data

    def duplicates(self, col=None):
        if col is None:
            dup = self.df.duplicated().sum()
        else:
            dup = self.df[col].duplicated().sum()
        return dup

    def describe_dataframe(self):
        return self.df.describe().T

    def unique_values(self, col):
        return self.df[col].unique()

    def info(self):
        return self.df.info()

    def skew_kurt(self, col=None):

        if col is None:
            kurt = self.df.kurt()
            skew = self.df.skew()

            axis = list(skew.axes[0])
            skew_log = self.dataset.transformation(copy.deepcopy(self.df[axis]), TransformersType.LOG).skew()
            skew_sqrt = self.dataset.transformation(copy.deepcopy(self.df[axis]), TransformersType.SQRT).skew()
            skew_box_cox = self.dataset.transformation(copy.deepcopy(self.df[axis]), TransformersType.BOX_COX).skew()

            skew_box_cox = np.array(skew_box_cox)
            skew_box_cox_dic = {}
            for ax, val in zip(axis, skew_box_cox):
                skew_box_cox_dic[ax] = val

            skew_box_cox = pd.Series(skew_box_cox_dic)

            kurt_log = self.dataset.transformation(copy.deepcopy(self.df[axis]), TransformersType.LOG).kurt()
            kurt_sqrt = self.dataset.transformation(copy.deepcopy(self.df[axis]), TransformersType.SQRT).kurt()
            kurt_box_cox = self.dataset.transformation(copy.deepcopy(self.df[axis]), TransformersType.BOX_COX).kurt()

            kurt_box_cox = np.array(kurt_box_cox)
            kurt_box_cox_dic = {}
            for ax, val in zip(axis, kurt_box_cox):
                kurt_box_cox_dic[ax] = val

            kurt_box_cox = pd.Series(kurt_box_cox_dic)

            return pd.concat([skew, skew_log, skew_sqrt, skew_box_cox, kurt, kurt_log, kurt_sqrt, kurt_box_cox], axis=1,
                             keys=['skew', 'skew log', 'skew sqrt', 'skew box cox ',
                                   'kurt', 'kurt log', 'kurt sqrt', 'kurt box cox']).sort_values(
                by=['skew'],
                ascending=False)
        else:
            if is_numeric_dtype(self.df[col]):
                kurt = self.df[col].kurt()
                skew = self.df[col].skew()
                return {
                    'skew': skew,
                    'kurt': kurt
                }
            else:
                return '{} is categorical feature'.format(col)

    def quantiles(self, col):
        if col is None:
            return self.df.quantile([.1, .25, .5, .75], axis=0).T
        else:
            if is_numeric_dtype(self.df[col]):
                new_df = self.df[col].quantile([.1, .25, .5, .75]).reset_index()
                new_df.columns = ['quantile', 'values']
                return new_df
            else:
                return '{} is categorical feature'.format(col)

    @property
    def target_col(self):
        return self._dataset.target_col

    @property
    def target(self):
        return self._dataset.target

    @property
    def df(self):
        return self._dataset.df

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
