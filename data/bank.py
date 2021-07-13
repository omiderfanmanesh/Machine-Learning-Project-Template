#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.


from data.based.based_dataset import BasedDataset
from data.based.file_types import FileTypes


class Bank(BasedDataset):
    def __init__(self, cfg):
        super(Bank, self).__init__(cfg=cfg, dataset_type=FileTypes.CSV)

    def age(self):
        self.df.loc[self.df['age'] <= 32, 'age'] = 1
        self.df.loc[(self.df['age'] > 32) & (self.df['age'] <= 47), 'age'] = 2
        self.df.loc[(self.df['age'] > 47) & (self.df['age'] <= 70), 'age'] = 3
        self.df.loc[(self.df['age'] > 70) & (self.df['age'] <= 98), 'age'] = 4

    def job(self):
        pass

    def marital(self):
        pass

    def education(self):
        pass

    def default(self):
        pass

    def balance(self):
        pass

    def housing(self):
        pass

    def loan(self):
        pass

    def contact(self):
        pass

    def day(self):
        pass

    def month(self):
        pass

    def duration(self):
        self.df.loc[self.df['duration'] <= 102, 'duration'] = 1
        self.df.loc[(self.df['duration'] > 102) & (self.df['duration'] <= 180), 'duration'] = 2
        self.df.loc[(self.df['duration'] > 180) & (self.df['duration'] <= 319), 'duration'] = 3
        self.df.loc[(self.df['duration'] > 319) & (self.df['duration'] <= 644.5), 'duration'] = 4
        self.df.loc[self.df['duration'] > 644.5, 'duration'] = 5

    def campaign(self):
        pass

    def pdays(self):
        pass

    def previous(self):
        pass

    def poutcome(self):
        self.df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1, 2, 3], inplace=True)

    def y(self):
        pass
