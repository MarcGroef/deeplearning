#!/usr/bin/python
import numpy as np
from trainer import Trainer

class Experiment:
    def __init__(self):
        self.experimentsToDo = ['control', 'batchnorm', 'dropout', 'l1', 'l2']
        self.nExperiments = 1
        self.results = {}


    def runExperiments(self):
        for experiment in self.experimentsToDo:
            self.results[experiment] = {}
            self.results[experiment]['train_acc'] = []
            self.results[experiment]['train_loss'] = []
            self.results[experiment]['val_acc'] = []
            self.results[experiment]['val_loss'] = []

            for expIdx in range(self.nExperiments):
                trainer = Trainer(experiment, expIdx)

                tr_acc, tr_loss, val_acc, val_loss = trainer.train()
                self.results[experiment]['train_acc'].append(tr_acc)
                self.results[experiment]['train_loss'].append(tr_loss)
                self.results[experiment]['val_acc'].append(val_acc)
                self.results[experiment]['val_loss'].append(val_loss)
                trainer.storeStats(expIdx)
                del trainer

if __name__ == "__main__":
    exp = Experiment()
    exp.runExperiments()
