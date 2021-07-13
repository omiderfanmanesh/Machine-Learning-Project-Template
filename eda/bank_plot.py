#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from data.based import TransformersType
from eda.based import BasedPlot


class BankPlots(BasedPlot):
    def __init__(self, cfg, dataset):
        super(BankPlots, self).__init__(dataset=dataset, cfg=cfg)

    def age(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def job(self):
        self.category_count(col='job', lbl_rotation=90)

    def marital(self):
        self.category_count(col='marital')

    def education(self):
        self.category_count(col='education')
        self.bar(x='education', y='age')

    def default(self):
        self.category_count(col='default')

    def balance(self):
        self.box_by_col(col='balance')
        self.box_by_col(col='balance', trans=TransformersType.LOG)

        self.dist_by_col(col='balance')
        self.dist_by_col(col='balance', trans=TransformersType.LOG)

    def housing(self):
        self.category_count(col='housing')

    def loan(self):
        self.category_count(col='loan')

    def contact(self):
        self.category_count(col='contact')

    def day(self):
        self.box_by_col(col='day')
        self.box_by_col(col='day', trans=TransformersType.LOG)

        self.dist_by_col(col='day')
        self.dist_by_col(col='day', trans=TransformersType.LOG)

        self.violin_by_col(col='day')
        self.violin_by_col(col='day', trans=TransformersType.LOG)

    def month(self):
        self.category_count(col='month')

    def duration(self):
        self.box_by_col(col='duration')
        self.box_by_col(col='duration', trans=TransformersType.LOG)

        self.dist_by_col(col='duration')
        self.dist_by_col(col='duration', trans=TransformersType.LOG)

        self.violin_by_col(col='duration')
        self.violin_by_col(col='duration', trans=TransformersType.LOG)

    def campaign(self):
        self.box_by_col(col='campaign')
        self.box_by_col(col='campaign', trans=TransformersType.LOG)

        self.dist_by_col(col='campaign')
        self.dist_by_col(col='campaign', trans=TransformersType.LOG)

        self.violin_by_col(col='campaign')
        self.violin_by_col(col='campaign', trans=TransformersType.LOG)

    def pdays(self):
        self.box_by_col(col='pdays')
        self.box_by_col(col='pdays', trans=TransformersType.LOG)

        self.dist_by_col(col='pdays')
        self.dist_by_col(col='pdays', trans=TransformersType.LOG)

        self.violin_by_col(col='pdays')
        self.violin_by_col(col='pdays', trans=TransformersType.LOG)

    def previous(self):
        self.box_by_col(col='previous')
        self.box_by_col(col='previous', trans=TransformersType.LOG)

        self.dist_by_col(col='previous')
        self.dist_by_col(col='previous', trans=TransformersType.LOG)

        self.violin_by_col(col='previous')
        self.violin_by_col(col='previous', trans=TransformersType.LOG)

    def poutcome(self):
        self.category_count(col='poutcome')

    def y(self):
        self.category_count(col='y')

    def resample(self, encoder, scaler):
        self.dataset.df[self.dataset.target_col] = encoder.custom_encoding(self.dataset.df, col=self.cfg.DATASET.TARGET,
                                                                           encode_type=self.cfg.ENCODER.Y)
        if encoder is None:
            _data = self.dataset.select_columns(data=self.dataset.df)
        else:
            # convert categorical features to integer
            _data = encoder.do_encode(data=self.dataset.df, y=self.dataset.targets.values)

        _y = _data[self.dataset.target_col]
        _X = _data.drop([self.dataset.target_col], axis=1)

        # change the scale of data
        if scaler is not None:
            _X = scaler.do_scale(data=_X)

        # if you set the resampling strategy, it will balance your data based on your strategy
        counter = Counter(_y)
        print(f"Before sampling {counter}")
        _X_resample, _y_resample = self.dataset.resampling(X=_X, y=_y)
        counter = Counter(_y_resample)
        print(f"After sampling {counter}")

        print("plotting origin values...")
        _X['y'] = _y
        sns.pairplot(_X, hue="y", height=2.5)
        plt.show()
        print("plotting resampled values...")
        _X_resample['y'] = _y_resample
        sns.pairplot(_X_resample, hue="y", height=2.5)
        plt.show()
