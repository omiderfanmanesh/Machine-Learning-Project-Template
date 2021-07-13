#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA as skl_pca

sns.set()


class PCA:
    def __init__(self, cfg):
        self.n_components = cfg.PCA.N_COMPONENTS
        self.pca = skl_pca(n_components=self.n_components, random_state=cfg.BASIC.RAND_STATE)

    def apply(self, data=None, y=None,
              X_train=None, X_test=None):
        """
        apply pca to data
        :param y:
        :param X_test:
        :param X_train:
        :param data:
        :return: dataframe of pca components
        """
        if data is not None and y is not None:
            _components = self.pca.fit_transform(data)
            print('Explained variance: %.4f' % self.pca.explained_variance_ratio_.sum())
            print('Individual variance contributions:')
            for j in range(self.pca.n_components_):
                print(f"PC({j}): {self.pca.explained_variance_ratio_[j]}")

            _columns = ['pc' + str(i + 1) for i in range(self.pca.n_components_)]
            _columns.append('y')

            y = np.reshape(y.values, (y.values.shape[0], -1)).copy()
            _components = np.concatenate((_components, y), axis=1)
            _pca_df = pd.DataFrame(data=_components
                                   , columns=_columns)

            return _pca_df

        elif X_train is not None and X_test is not None:
            _components = self.pca.fit(X=X_train)
            print('Explained variance: %.4f' % self.pca.explained_variance_ratio_.sum())
            print('Individual variance contributions:')
            for j in range(self.pca.n_components_):
                print(f"PC({j}): {self.pca.explained_variance_ratio_[j]:.4f}")
            _columns = ['pc' + str(i + 1) for i in range(self.pca.n_components_)]

            n_train = self.pca.transform(X=X_train)
            n_test = self.pca.transform(X=X_test)

            df_train = pd.DataFrame(data=n_train, columns=_columns)
            df_test = pd.DataFrame(data=n_test, columns=_columns)

            return df_train, df_test

    def plot(self, X, y):
        """
        scatter plot of pca components
        :param X:
        :param y:
        """
        X['y'] = y
        sns.pairplot(X, hue="y", height=2.5)
        plt.show()
