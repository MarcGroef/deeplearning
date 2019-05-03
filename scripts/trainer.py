from network import Network
from dataset import Dataset

class Trainer():
    def __init__(self):
        self.net = Network()
        self.data = Dataset()
        
    def train(self):
        self.net.train(self.data.trainImages, self.data.trainLabels, self.data.trainImages, self.data.trainLabels, 10, 10)
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()