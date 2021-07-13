#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from sklearn.svm import SVC, SVR

from model.based import BasedModel
from model.based import TaskMode


class SVM(BasedModel):
    def __init__(self, cfg):
        super(SVM, self).__init__(cfg=cfg)
        self._task_mode = cfg.BASIC.TASK_MODE

        if self._task_mode == TaskMode.CLASSIFICATION:
            self._params = {
                'C': cfg.SVM.C,
                'kernel': cfg.SVM.KERNEL,
                'degree': cfg.SVM.DEGREE,
                'gamma': cfg.SVM.GAMMA,
                'coef0': cfg.SVM.COEF0,
                'shrinking': cfg.SVM.SHRINKING,
                'probability': cfg.SVM.PROBABILITY,
                'tol': cfg.SVM.TOL,
                'cache_size': cfg.SVM.CACHE_SIZE,
                'class_weight': cfg.SVM.CLASS_WEIGHT,
                'verbose': cfg.SVM.VERBOSE,
                'max_iter': cfg.SVM.MAX_ITER,
                'decision_function_shape': cfg.SVM.DECISION_FUNCTION_SHAPE,
                'break_ties': cfg.SVM.BREAK_TIES,
                'random_state': cfg.SVM.RANDOM_STATE

            }
            self.model = SVC(**self._params)
            self.name = cfg.SVM.NAME
            for _k in cfg.SVM.HYPER_PARAM_TUNING:
                _param = cfg.SVM.HYPER_PARAM_TUNING[_k]

                if _param is not None:
                    _param = [*_param]
                    if _k is 'C':
                        self.fine_tune_params[_k] = [*_param]
                    else:
                        self.fine_tune_params[_k.lower()] = [*_param]

        elif self._task_mode == TaskMode.REGRESSION:
            self._params = {
                'kernel': cfg.SVR.KERNEL,
                'degree': cfg.SVR.DEGREE,
                'gamma': cfg.SVR.GAMMA,
                'coef0': cfg.SVR.COEF0,
                'tol': cfg.SVR.TOL,
                'C': cfg.SVR.C,
                'epsilon': cfg.SVR.EPSILON,
                'shrinking': cfg.SVR.SHRINKING,
                'cache_size': cfg.SVR.CACHE_SIZE,
                'verbose': cfg.SVR.VERBOSE,
                'max_iter': cfg.SVR.MAX_ITER

            }
            self.model = SVR(**self._params)
            self.name = cfg.SVR.NAME
            for _k in cfg.SVR.HYPER_PARAM_TUNING:
                _param = cfg.SVR.HYPER_PARAM_TUNING[_k]
                if _param is not None:
                    _param = [*_param]
                    if _k is 'C':
                        self.fine_tune_params[_k] = [*_param]
                    else:
                        self.fine_tune_params[_k.lower()] = [*_param]
