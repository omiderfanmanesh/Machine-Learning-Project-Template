#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.


import warnings

from configs import cfg
from data import load
from data.preprocessing import Encoders, Scalers
from eda.bank_analyser import BankAnalyzer
from eda.bank_plot import BankPlots
from engine.analyser import do_analysis

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    bank = load(cfg)
    bank.load_dataset()
    bank.drop_cols()

    encoder = Encoders(cdg=cfg)
    scaler = Scalers(cfg=cfg)

    analyzer = BankAnalyzer(dataset=bank, cfg=cfg)
    plots = BankPlots(dataset=bank, cfg=cfg)

    do_analysis(dataset=bank, analyzer=analyzer, plots=plots, encoder=encoder, scaler=scaler)


if __name__ == '__main__':
    main()
