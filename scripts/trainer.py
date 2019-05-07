#!/usr/bin/python
from network import Network
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, experimentType, exp_idx):
        self.validExperimentTypes = ['control', 'batchnorm', 'dropout', 'l2', 'l1']
        assert(experimentType in self.validExperimentTypes), ("Invalid experiment type.. Please choose from:\n" + str(self.validExperimentTypes))
        self.experimentType = experimentType

        self.net = Network(experimentType)
        self.data = Dataset(10, exp_idx)

    def train(self):
<<<<<<< HEAD
        history = self.net.train(self.data.trainImages, self.data.trainLabels, self.data.testImages, self.data.testLabels, 200, 20)
=======
        history = self.net.train(self.data.trainImages(), self.data.trainLabels(), self.data.valImages(), self.data.valLabels(), 200, 10)
>>>>>>> 172a6af21eb7856cadb7c8f680b4f0e3ddfdd551
        self.val_acc = history['val_acc']
        self.val_loss = history['val_loss']
        self.train_acc = history['acc']
        self.train_loss = history['loss']
        return (self.train_acc, self.train_loss, self.val_acc, self.val_loss)

    def storeStats(self, experimentIdx):
        np.save("../data/" + self.experimentType + "_train_acc_" + str(experimentIdx), self.train_acc)
        np.save("../data/" + self.experimentType + "_train_loss_" + str(experimentIdx), self.train_loss)
        np.save("../data/" + self.experimentType + "_val_acc_" + str(experimentIdx), self.val_acc)
        np.save("../data/" + self.experimentType + "_val_loss_" + str(experimentIdx), self.val_loss)

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
