import numpy as np
import tensorflow as tf


class Dataset:
    
    def __init__(self):
        (self.trainImages, self.trainLabels), (self.testImages, self.testLabels) = tf.keras.datasets.fashion_mnist.load_data()
        self.trainImages = self.trainImages.astype('float32') / 255
        self.testImages = self.testImages.astype('float32') / 255
        self.trainImages = np.expand_dims(self.trainImages, axis = -1)
        self.testImages = np.expand_dims(self.testImages, axis = -1)
        self.testLabels = tf.keras.utils.to_categorical(self.testLabels)
        self.trainLabels = tf.keras.utils.to_categorical(self.trainLabels)
if __name__ == "__main__":
    dataset = Dataset();
