#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from sklearn.linear_model import LogisticRegression as lg

from model.based import BasedModel, TaskMode


class LogisticRegression(BasedModel):
    def __init__(self, cfg):
        super(LogisticRegression, self).__init__(cfg=cfg)

        self.use_for_feature_importance = True
        self._task_mode = cfg.BASIC.TASK_MODE

        self._params = {

            'penalty': cfg.LOGISTIC_REGRESSION.PENALTY,
            'dual': cfg.LOGISTIC_REGRESSION.DUAL,
            'tol': cfg.LOGISTIC_REGRESSION.TOL,
            'C': cfg.LOGISTIC_REGRESSION.C,
            'fit_intercept': cfg.LOGISTIC_REGRESSION.FIT_INTERCEPT,
            'intercept_scaling': cfg.LOGISTIC_REGRESSION.INTERCEPT_SCALING,
            'class_weight': cfg.LOGISTIC_REGRESSION.CLASS_WEIGHT,
            'random_state': cfg.LOGISTIC_REGRESSION.RANDOM_STATE,
            'solver': cfg.LOGISTIC_REGRESSION.SOLVER,

            'max_iter': cfg.LOGISTIC_REGRESSION.MAX_ITER,
            'multi_class': cfg.LOGISTIC_REGRESSION.MULTI_CLASS,
            'verbose': cfg.LOGISTIC_REGRESSION.VERBOSE,
            'warm_start': cfg.LOGISTIC_REGRESSION.WARM_START,
            'n_jobs': cfg.LOGISTIC_REGRESSION.N_JOBS,
            'l1_ratio': cfg.LOGISTIC_REGRESSION.L1_RATIO,

        }
        self.name = cfg.LOGISTIC_REGRESSION.NAME

        if self._task_mode == TaskMode.CLASSIFICATION:
            self.model = lg(**self._params)

            for _k in cfg.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING:
                _param = cfg.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING[_k]

                if _param is not None:
                    _param = [*_param]
                    if _k is 'C':
                        self.fine_tune_params[_k] = [*_param]
                    else:
                        self.fine_tune_params[_k.lower()] = [*_param]
        else:
            print(f"this model can not be used for regression task")
