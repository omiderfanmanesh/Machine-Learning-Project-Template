#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from model.based import BasedModel, TaskMode


class DecisionTree(BasedModel):
    def __init__(self, cfg):
        super(DecisionTree, self).__init__(cfg=cfg)
        self.use_for_feature_importance = True
        self._task_mode = cfg.BASIC.TASK_MODE

        self._params = {
            'criterion': cfg.DECISION_TREE.CRITERION,
            'splitter': cfg.DECISION_TREE.SPLITTER,
            'max_depth': cfg.DECISION_TREE.MAX_DEPTH,
            'min_samples_split': cfg.DECISION_TREE.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': cfg.DECISION_TREE.MIN_SAMPLES_LEAF,
            'min_weight_fraction_leaf': cfg.DECISION_TREE.MIN_WEIGHT_FRACTION_LEAF,
            'max_features': cfg.DECISION_TREE.MAX_FEATURES,
            'random_state': cfg.DECISION_TREE.RANDOM_STATE,
            'max_leaf_nodes': cfg.DECISION_TREE.MAX_LEAF_NODES,
            'min_impurity_decrease': cfg.DECISION_TREE.MIN_IMPURITY_DECREASE,
            'min_impurity_split': cfg.DECISION_TREE.MIN_IMPURITY_SPLIT,
            'class_weight': cfg.DECISION_TREE.CLASS_WEIGHT,
            'presort': cfg.DECISION_TREE.PRESORT,
            'ccp_alpha': cfg.DECISION_TREE.CCP_ALPHA,
        }

        if self._task_mode == TaskMode.CLASSIFICATION:
            self.model = DecisionTreeClassifier(**self._params)
        elif self._task_mode == TaskMode.REGRESSION:
            self.model = DecisionTreeRegressor(**self._params)

        self.name = cfg.DECISION_TREE.NAME

        for _k in cfg.DECISION_TREE.HYPER_PARAM_TUNING:
            _param = cfg.DECISION_TREE.HYPER_PARAM_TUNING[_k]

            if _param is not None:
                _param = [*_param]
                self.fine_tune_params[_k.lower()] = [*_param]
