import numpy as np
import tensorflow as tf

class Dataset:
    
    def __init__(self):
        (self.trainImages, self.trainLabels), (self.testImages, self.testLabels) = tf.keras.datasets.fashion_mnist.load_data()
        
        
if __name__ == "__main__":
    dataset = Dataset();
