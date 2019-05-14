#!/usr/bin/python
import numpy as np
from trainer import Trainer

class Experiment:
    def __init__(self):
        #self.experimentsToDo = ['control', 'batchnorm', 'dropout', 'l1', 'l2']
        # self.experimentsToDo = [{'experiment' : 'l2'}]
        self.experimentsToDo = [{'exp_type' : 'l2', 'l2_scalar' : 0.01}, {'exp_type' : 'l2', 'l2_scalar' : 0.007}, {'exp_type' : 'l2', 'l2_scalar' : 0.004}, {'exp_type' : 'l2', 'l2_scalar' : 0.001}, {'exp_type' : 'l2', 'l2_scalar' : 0.0007}]

        self.nExperiments = 10
        self.results = {}


    def runExperiments(self):
        for exp in self.experimentsToDo:
            experiment = exp['exp_type']
            self.results[experiment] = {}
            self.results[experiment]['train_acc'] = []
            self.results[experiment]['train_loss'] = []
            self.results[experiment]['val_acc'] = []
            self.results[experiment]['val_loss'] = []

            for expIdx in range(self.nExperiments):
                trainer = Trainer(experiment, expIdx, **exp)

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
