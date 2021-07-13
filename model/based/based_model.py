#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from pprint import pprint
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dtreeviz.trees import dtreeviz  # remember to load the package
# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback

from .metric_types import MetricTypes
from .tuning_mode import TuningMode


class BasedModel:
    def __init__(self, cfg):
        self.model = None
        self._metric_function = cfg.EVALUATION.METRIC
        self._fold = cfg.MODEL.K_FOLD
        self.name = None
        self.use_for_feature_importance = False
        self.fine_tune_params = {}
        self._confusion_matrix = cfg.EVALUATION.CONFUSION_MATRIX

    def train(self, X_train, y_train):
        """
        train the model

        :param X_train:
        :param y_train:
        :return:
        """
        print('start training...')
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test, target_labels=None, normalize=None):
        """
        evaluate the model based on a metric
        :param X_test: test set
        :param y_test: test targets
        :param target_labels: distinct target values in list
        :param normalize: it is for confusion matrix
        """
        print('evaluation...')
        y_pred = self.model.predict(X_test)
        score = self.metric(y_test, y_pred)
        print(f'score is {score:.3f}')

        if self._confusion_matrix:
            plot_confusion_matrix(self.model, X_test, y_test, cmap=plt.cm.Blues,
                                  display_labels=target_labels,
                                  normalize=normalize)
            plt.show()

    def metric(self, y_true=None, y_pred=None):
        """
        initialize the metric for evaluation the model

        :param y_true:
        :param y_pred:
        :return: metric obj or name
        """
        metric_type = self._metric_function

        if y_pred is None and y_true is None:
            if metric_type == MetricTypes.F1_SCORE_BINARY:
                return 'f1'
            elif metric_type == MetricTypes.F1_SCORE_MICRO:
                return 'f1_micro'
            elif metric_type == MetricTypes.F1_SCORE_MACRO:
                return 'f1_macro'
            elif metric_type == MetricTypes.F1_SCORE_WEIGHTED:
                return 'f1_weighted'
            elif metric_type == MetricTypes.F1_SCORE_SAMPLE:
                return 'f1_samples'
            elif metric_type == MetricTypes.PRECISION:
                return 'precision'
            elif metric_type == MetricTypes.RECALL:
                return 'recall'
            elif metric_type == MetricTypes.ACCURACY:
                return 'accuracy'
        else:
            if metric_type == MetricTypes.F1_SCORE_BINARY:
                return f1_score(y_true, y_pred, average="binary")
            elif metric_type == MetricTypes.F1_SCORE_MICRO:
                return f1_score(y_true, y_pred, average="micro")
            elif metric_type == MetricTypes.F1_SCORE_MACRO:
                return f1_score(y_true, y_pred, average="macro")
            elif metric_type == MetricTypes.F1_SCORE_WEIGHTED:
                return f1_score(y_true, y_pred, average="weighted")
            elif metric_type == MetricTypes.F1_SCORE_SAMPLE:
                return f1_score(y_true, y_pred, average="sample")
            elif metric_type == MetricTypes.PRECISION:
                return precision_score(y_true, y_pred)
            elif metric_type == MetricTypes.RECALL:
                return recall_score(y_true, y_pred)
            elif metric_type == MetricTypes.ACCURACY:
                return accuracy_score(y_true, y_pred)

    def hyper_parameter_tuning(self, X, y, title='', method=TuningMode.GRID_SEARCH):
        """
        apply hyper parameter tuning
        :param X:
        :param y:
        :param title:
        :param method:
        :return:
        """
        opt = None
        callbacks = None
        if self.fine_tune_params:
            if method == TuningMode.GRID_SEARCH:
                opt = GridSearchCV(estimator=self.model, param_grid=self.fine_tune_params, cv=3, n_jobs=-1, verbose=3,
                                   scoring=self.metric())
            elif method == TuningMode.BAYES_SEARCH:
                opt = BayesSearchCV(self.model, self.fine_tune_params)
                callbacks = [VerboseCallback(100), DeadlineStopper(60 * 10)]

            best_params = self.report_best_params(optimizer=opt, X=X, y=y, title=title,
                                                  callbacks=callbacks)
            return best_params
        else:
            print('There are no params for tuning')

    def feature_importance(self, features=None):
        """
        detect important features for a model

        :param features: column names, it will be used for printing the columns
        """
        if self.use_for_feature_importance:
            if hasattr(self.model, 'coef_'):
                importance = self.model.coef_[0]
            else:
                importance = self.model.feature_importances_
            importance = np.array(importance).reshape((-1, 1))

            # summarize feature importance
            for i, v in enumerate(importance):
                if features:
                    print('Feature(%0d): %0s, Score: %.5f' % (i, str(features[i]), v))
                else:
                    print('Feature(%0s): Score: %.5f' % (str(i), v))

            fs = pd.DataFrame(data=importance.T, columns=features)
            ax = sns.barplot(data=fs)
            plt.gcf().set_size_inches(11, 9)
            plt.xticks(rotation=90)
            plt.title(f'Feature importance by using the model of {self.name}')
            plt.show()
        else:
            print(f" The model of {self.name} can not be used for estimating the importance of features")

    def report_best_params(self, optimizer, X, y, title, callbacks=None):
        """
        A wrapper for measuring time and performances of different optimizers

        optimizer = a sklearn or a skopt optimizer
        X = the training set
        y = our target
        title = a string label for the experiment
        """
        start = time()
        if callbacks:
            optimizer.fit(X, y, callback=callbacks)
        else:
            optimizer.fit(X, y)

        d = pd.DataFrame(optimizer.cv_results_)
        best_score = optimizer.best_score_
        best_score_std = d.iloc[optimizer.best_index_].std_test_score
        best_params = optimizer.best_params_
        if best_params:
            print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
                   + u"\u00B1" + " %.3f") % (time() - start,
                                             len(optimizer.cv_results_['params']),
                                             best_score,
                                             best_score_std))
            print('Best parameters:')
            pprint(best_params)
            print()
        else:
            print('There are no params provided')

        return best_params

    def plot_tree(self, X, y, target_name, feature_names, class_names):
        """
        plot the decision trees. Note that it will be work just for decision tree classifier
        :param X:
        :param y:
        :param target_name:
        :param feature_names:
        :param class_names:
        """
        viz = dtreeviz(self.model, X, y,
                       target_name=target_name,
                       feature_names=feature_names,
                       class_names=list(class_names))
        viz.save("decision_tree.svg")
        viz.view()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def fine_tune_params(self):
        return self._fine_tune_params

    @fine_tune_params.setter
    def fine_tune_params(self, value):
        self._fine_tune_params = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
