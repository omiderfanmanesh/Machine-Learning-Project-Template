#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from data.bank import Bank


def load(cfg):
    bank = Bank(cfg=cfg)
    return bank
