#!/usr/bin/bash
import numpy as np
from trainer import Trainer

if __name__ == "__main__":
    mean = 0.0
    nExps = 10
    for expIdx in range(nExps):
        exp = {'exp_type' : 'dropout', 'fc_dropout_rate' : 0.4}
        trainer = Trainer('dropout', exp)
        trainer.train()
        mean += trainer.test()
    mean /= nExps
    print "Result Dropout: ": mean