#!/usr/bin/python
import numpy as np
from trainer import Trainer

class Experiment:
    def __init__(self):        
        # self.experimentsToDo = [{'exp_type' : 'control'}, {'exp_type' : 'batchnorm'}]
        # self.experimentsToDo = [{'exp_type' : 'l2', 'l2_scalar' : param} for param in np.arange(0, 0.003, 0.00025)]
        self.experimentsToDo = [{'exp_type' : 'l1', 'l1_scalar' : param} for param in np.arange(0, 0.003, 0.00025)]
        self.experimentsToDo = [{'exp_type' : 'dropout', 'fc_dropout_rate' : param*.1} for param in range(10)]

        self.nExperiments = 5
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
                exp['nExperiments'] = self.nExperiments
                exp['experiment_index'] = expIdx
                trainer = Trainer(experiment, **exp)

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
