import numpy as np
import tensorflow as tf

class Dataset:
    
    def __init__(self):
        (self.trainImages, self.trainLabels), (self.testImages, self.testLabels) = tf.keras.datasets.fashion_mnist.load_data()
        self.trainImages = self.trainImages.astype('float32') / 255
        self.testImages = self.testImages.astype('float32') / 255
        
if __name__ == "__main__":
    dataset = Dataset();
