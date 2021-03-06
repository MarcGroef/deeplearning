#!/usr/bin/python
from network import Network, PARAMETER_EXPERIMENTS
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os


class Trainer():
    def __init__(self, experimentType, **kwargs):
        self.validExperimentTypes = ['control', 'batchnorm', 'dropout', 'l2', 'l1']
        assert(experimentType in self.validExperimentTypes), ("Invalid experiment type.. Please choose from:\n" + str(self.validExperimentTypes))
        self.experimentType = experimentType

        self.kwargs = kwargs
        self.net = Network(experimentType, **kwargs)
        self.data = Dataset(kwargs['nExperiments'], kwargs['experiment_index'])
        self.save_loc = '../data/'

    def train(self):
        history = self.net.train(self.data.trainImages(), self.data.trainLabels(), self.data.valImages(), self.data.valLabels(), 512, epochs=25)
        self.val_acc = history['val_acc']
        self.val_loss = history['val_loss']
        self.train_acc = history['acc']
        self.train_loss = history['loss']
        return (self.train_acc, self.train_loss, self.val_acc, self.val_loss)
    
    def test(self):
        res = self.net.test(self.data.testImages(), self.data.testLabels())
        return res

    def storeStats(self, experimentIdx):
        if not os.path.exists(self.save_loc):
            os.makedirs(self.save_loc)

        # Tag parameters being tested
        tag = "".join([param + '=' + str(self.kwargs[param]) for param in PARAMETER_EXPERIMENTS if param in self.kwargs])
        tag = '_' + tag if tag != "" else ""

        np.save(self.save_loc + self.experimentType + tag + "_train_acc_" + str(experimentIdx), self.train_acc)
        np.save(self.save_loc + self.experimentType + tag + "_train_loss_" + str(experimentIdx), self.train_loss)
        np.save(self.save_loc + self.experimentType + tag + "_val_acc_" + str(experimentIdx), self.val_acc)
        np.save(self.save_loc + self.experimentType + tag + "_val_loss_" + str(experimentIdx), self.val_loss)

    def printStats(self):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.val_acc)), self.val_acc, label='validation')
        ax.plot(range(len(self.train_acc)), self.train_acc, label='train')
        ax.legend(loc='upper left')
        ax.set(xlabel = "epoch", ylabel="acc")
        ax.grid()
        plt.savefig("results_" + self.experimentType +".png")
        #plt.show()

    def setExperimentType(self, experimentType):
        assert(experimentType in self.validExperimentTypes), ("Invalid experiment type.. Please choose from:\n" + str(self.validExperimentTypes))
        self.experimentType = experimentType

        self.net = Network(experimentType)

if __name__ == "__main__":

    trainer = Trainer('control')
    trainer.train()
    trainer.printStats()
    trainer.storeStats()

    trainer = Trainer('batchnorm')
    trainer.train()
    trainer.printStats("test_function")
    trainer.storeStats()
