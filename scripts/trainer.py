#!/usr/bin/python
from network import Network
from dataset import Dataset

import matplotlib.pyplot as plt


class Trainer():
    def __init__(self):
        self.net = Network()
        self.data = Dataset()
        
    def train(self):
        history = self.net.train(self.data.trainImages, self.data.trainLabels, self.data.trainImages, self.data.trainLabels, 2000, 10)
        self.val_acc = history['val_acc']
        self.val_loss = history['val_loss']
        self.train_acc = history['acc']
        self.train_loss = history['loss']

    def outputStats(self, id):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.val_acc)), self.val_acc, label='validation')
        ax.plot(range(len(self.train_acc)), self.train_acc, label='train')
        ax.legend(loc='upper left')
        ax.set(xlabel = "epoch", ylabel="acc")
        ax.grid()
        plt.savefig("results_" + id +".png")
        plt.show()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.outputStats('test')