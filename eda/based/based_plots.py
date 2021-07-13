#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.


from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_string_dtype

from data.based import TransformersType, BasedDataset

sns.set_theme(style="white")


class BasedPlot:
    def __init__(self, dataset: BasedDataset, cfg):
        self._dataset = dataset
        self._df = dataset.df
        self.cfg = cfg

    def category_count(self, col, lbl_rotation=None):

        sns.catplot(x=col, hue=self.target, kind="count", data=self.df)
        fig = plt.gcf()
        fig.set_size_inches(8, 10)
        if lbl_rotation is not None:
            plt.xticks(rotation=lbl_rotation)
        plt.show()

    def bar(self, x, y):
        sns.barplot(x=x, y=y, hue=self.target, data=self.df)
        plt.show()

    def strip(self, x, y):
        sns.stripplot(x=x, y=y, hue=self.target, data=self.df)
        plt.show()

    def kernel_density_estimation(self, x, y):
        g = sns.kdeplot(
            data=self.df,
            x=x,
            y=y,
            hue=self.target,
            thresh=.1,
        )
        plt.show()

    def linear_regression(self, x, y, joint=False):
        if joint:
            g = sns.jointplot(x=x, y=y, data=self.df, kind="reg")
        else:
            g = sns.lmplot(
                data=self.df,
                x=x, y=y, hue=self.target,
                height=5
            )
        g.fig.set_figheight(6)
        g.fig.set_figwidth(10)
        # Use more informative axis labels than are provided by default
        g.set_axis_labels("{}".format(x), "{}".format(y))
        plt.show()

    def rel(self, x, y):

        g = sns.relplot(x=x, y=y, hue=self.target, style=self.target,
                        data=self.df)
        g.fig.set_figheight(6)
        g.fig.set_figwidth(10)
        plt.show()

    def pair(self):
        plt.figure(figsize=(15, 8))
        sns.pairplot(data=self.df, hue=self.target, corner=True)
        plt.show()

    def __scatter(self, df, first=None, second=None, features=None, by=None, trans=None):
        if first is None and second is None and features is not None:
            g = sns.PairGrid(df, vars=features, hue=self.target, corner=True)
            g.map_diag(sns.histplot)
            g.map_offdiag(sns.scatterplot)
        else:
            if by is None:
                by = self.target
            sns.scatterplot(data=df, x=df[first], y=df[second], hue=by)
        plt.show()

    def __feature_distribution(self, df, col, is_numerical=True, trans=None):
        df[col] = self.dataset.transform(df[col], trans)
        if is_numerical:
            g = sns.FacetGrid(df, col=self.target, hue=self.target)
            g.map(sns.histplot, col)
        else:
            sns.countplot(data=df, x=col)
        plt.show()

    def numerical_features_distribution(self, trans=None):
        features = self.dataset.numerical_features()
        for f in features:
            self.__feature_distribution(self.df.copy(), f, is_numerical=True, trans=trans)

    def categorical_features_distribution(self, trans=None):
        features = self.dataset.categorical_features()
        for f in features:
            self.__feature_distribution(self.df.copy(), f, is_numerical=False, trans=trans)

    def numerical_features_scatter(self, trans=None):
        features = self.dataset.numerical_features()
        self.__scatter(self.df.copy(), features=features, trans=trans)

    def scatter(self, first, second, by, trans=None):
        if is_numeric_dtype(self.df[first]) and is_numeric_dtype(self.df[second]):
            self.__scatter(self.df.copy(), first=first, second=second, by=by, trans=trans)

    def numerical_features_box(self, trans=None):
        features = self.dataset.numerical_features()
        for f in features:
            self.box_by_col(col=f, trans=trans)

    def numerical_features_violin(self, trans=None):
        features = self.dataset.numerical_features()
        for f in features:
            self.violin_by_col(col=f, title=None, trans=trans)

    def __dist(self, params, title):
        g = sns.displot(**params)
        g.fig.set_size_inches(15, 13)
        plt.xticks(rotation=90)
        plt.title(title)
        plt.show()

    def __dist_sub(self, df):
        sub = ceil(len(self.df.columns) / 3)
        fig, axes = plt.subplots(nrows=sub, ncols=3)
        axes = axes.flatten()
        for ax, col in zip(axes, df.columns):
            sns.displot(df[col], ax=ax)
        plt.show()

    def dist_numerical_columns(self):
        numerical = self.dataset.numerical_features()
        self.__dist_sub(self.df[numerical])

    def dist_by_col(self, col, title=None, trans=TransformersType.NONE):
        df = self.df.copy()
        if is_string_dtype(df[col]):
            print('transformation is not possible for sting data!!!')
            trans = TransformersType.NONE
        else:
            df[col] = self.dataset.transform(df[col], trans)

        params = {
            'data': df,
            'x': col,
            'color': "green",
            'kde': True,
            'bins': 120,
            'hue': self.target
        }
        if title is None:
            title = 'Dist Plot of {} with Transformation type of {}'.format(col.upper(), trans.name)

        self.__dist(params, title=title)

    def dist(self, col, title=None, trans=TransformersType.NONE):
        df = self.df.copy()
        if is_string_dtype(df[col]):
            print('transformation is not possible for sting data!!!')
            trans = TransformersType.NONE
        else:
            df[col] = self.dataset.transform(df[col], trans)

        params = {
            'data': df,
            'x': col,
            'color': "green",
            'kde': True,
            'bins': 120
        }
        if title is None:
            title = 'Dist Plot of {} with Transformation type of {}'.format(col.upper(), trans.name)

        self.__dist(params, title)

    def __box(self, params, title):
        plt.figure(figsize=(15, 8))
        sns.boxplot(**params).set_title(title)
        sns.despine(offset=10, trim=True)

        plt.show()

    def box_by_col(self, col, title=None, trans=TransformersType.NONE):
        df = self.df.copy()
        if is_numeric_dtype(df[col]):
            df[col] = self.dataset.transform(df[col], trans)
            params = {
                'data': df,
                'x': col,
                'orient': "h",
                'y': self.target
            }
            if title is None:
                title = 'Box Plot of {} with Transformation type of {}'.format(col.upper(), trans.name)

            self.__box(params, title=title)
        else:
            print('{} is not numerical'.format(col))

    def box(self, col, title=None, trans=TransformersType.NONE):
        df = self.df.copy()
        df[col] = self.dataset.transform(df[col], trans)
        if is_numeric_dtype(df[col]):
            df[col] = self.dataset.transform(df[col], trans)
            params = {
                'data': df,
                'x': col,
                'orient': "h"
            }

            if title is None:
                title = 'Box Plot of {} with Transformation type of {}'.format(col.upper(), trans.name)

            self.__box(params, title=title)
        else:
            print('{} is not numerical'.format(col))

    def __violin(self, params, title):
        sns.violinplot(**params)
        plt.show()

    def violin_by_col(self, col, title=None, trans=TransformersType.NONE):
        df = self.df.copy()
        if is_numeric_dtype(df[col]):
            df[col] = self.dataset.transform(df[col], trans)

            params = {
                'data': df,
                'x': self.target,
                'y': col,
                'hue': self.target
            }

            if title is None:
                title = 'Violin Plot of {} with Transformation type of {}'.format(col.upper(), trans.name)

            self.__violin(params, title=title)
        else:
            print('{} is not numerical'.format(col))

    def violin(self, col, title=None, trans=TransformersType.NONE):
        df = self.df.copy()
        if is_numeric_dtype(df[col]):
            df[col] = self.dataset.transform(df[col], trans)

            params = {
                'data': df,
                'y': col
            }

            if title is None:
                title = 'Violin Plot of {} with Transformation type of {}'.format(col.upper(), trans.name)

            self.__violin(params, title=title)
        else:
            print('{} is not numerical'.format(col))

    def corr(self, data, all_methods=True):
        if all_methods:
            for method in ["pearson", "spearman", "kendall"]:
                self.__corr(data=data, method=method)
        else:
            self.__corr(data=data)

    def __corr(self, data, method='pearson'):
        _corr = data.corr(method=method)
        _mask = np.triu(np.ones_like(_corr, dtype=bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        _cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        _heat_map = sns.heatmap(_corr, mask=_mask, cmap=_cmap, vmax=.3, center=0,
                                square=True, linewidths=.5, cbar_kws={"shrink": .5})
        _heat_map.set_title('Feature Correlation by {}'.format(method))
        plt.show()

    @property
    def target(self):
        return self.dataset.target_col

    @property
    def df(self):
        return self.dataset.df

    @property
    def dataset(self):
        return self._dataset
