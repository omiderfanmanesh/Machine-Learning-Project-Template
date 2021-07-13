#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

import numpy as np
import pandas as pd

from eda.based import BasedAnalyzer


class BankAnalyzer(BasedAnalyzer):
    def __init__(self, dataset, cfg):
        super(BankAnalyzer, self).__init__(dataset, cfg)

    def age(self):
        self.description(col='age')

    def job(self):
        self.description(col='job')

    def marital(self):
        self.description(col='marital')

    def education(self):
        self.description(col='education')

        pivot = self.df[['education', 'y']].groupby(['education', 'y']).size()
        print('Pivot age/education (min)')
        print(pivot)
        print()

        pivot = pd.pivot_table(data=self.df, values=['age'], index=['education'], columns=['y'], aggfunc=np.min)
        print('Pivot age/education (min)')
        print(pivot)
        print()

        pivot = pd.pivot_table(data=self.df, values=['age'], index=['education'], columns=['y'], aggfunc=np.max)
        print('Pivot age/education (max)')
        print(pivot)
        print()

        pivot = pd.pivot_table(data=self.df, values=['age'], index=['education'], columns=['y'], aggfunc=np.mean)
        print('Pivot age/education (mean)')
        print(pivot)
        print()

    def default(self):
        self.description(col='default')

    def balance(self):
        self.description(col='balance')

    def housing(self):
        self.description(col='housing')

    def loan(self):
        self.description(col='loan')

    def contact(self):
        self.description(col='contact')

    def day(self):
        self.description(col='day')

    def month(self):
        self.description(col='month')

    def duration(self):
        self.description(col='duration')

    def campaign(self):
        self.description(col='campaign')

    def pdays(self):
        self.description(col='pdays')

    def previous(self):
        self.description(col='previous')

    def poutcome(self):
        self.description(col='poutcome')

    def y(self):
        self.description(col='y')
