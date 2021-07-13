#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from configs import cfg
from data.loader import load
from data.preprocessing import Encoders, Scalers, PCA
from engine.trainer import do_fine_tune, do_cross_val, do_train
from model import DecisionTree, LogisticRegression, SVM, RandomForest
from model.based import Model
from model.based.tuning_mode import TuningMode
from utils import RuntimeMode


def main():
    bank = load(cfg)  # create dataset object instance
    bank.load_dataset()  # load data from csv file
    bank.age()  # convert the range of the age's values to [1,2,3,4]
    bank.duration()  # convert the range of the duration's values to [1,2,3,4,5]
    bank.drop_cols()  # drop columns

    model_selection = cfg.BASIC.MODEL  # select the model
    if model_selection == Model.SVM:
        model = SVM(cfg=cfg)
    elif model_selection == Model.DECISION_TREE:
        model = DecisionTree(cfg=cfg)
    elif model_selection == Model.RANDOM_FOREST:
        model = RandomForest(cfg=cfg)
    elif model_selection == Model.LOGISTIC_REGRESSION:
        model = LogisticRegression(cfg=cfg)

    encoder = Encoders(cdg=cfg)  # initialize Encoder object
    scaler = Scalers(cfg=cfg)  # initialize scaler object
    pca = None
    if cfg.BASIC.PCA:  # PCA object will be initialized if you set pca = True in configs file
        pca = PCA(cfg=cfg)

    runtime_mode = cfg.BASIC.RUNTIME_MODE  # mode that you want to run this code
    if runtime_mode == RuntimeMode.TRAIN:
        do_train(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler, pca=pca)
    elif runtime_mode == RuntimeMode.CROSS_VAL:
        do_cross_val(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler, pca=pca)
    elif runtime_mode == RuntimeMode.TUNING:
        do_fine_tune(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler,
                     method=TuningMode.GRID_SEARCH)
    if runtime_mode == RuntimeMode.FEATURE_IMPORTANCE:
        do_train(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler, pca=pca, feature_importance=True)


if __name__ == '__main__':
    main()
