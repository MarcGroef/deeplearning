import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

# Set dataset seed
np.random.seed()

class SingletonDecorator:
    def __init__(self,klass):
        self.klass = klass
        self.instance = None
    def __call__(self,*args,**kwds):
        if self.instance == None:
            self.instance = self.klass(*args,**kwds)
        return self.instance

@SingletonDecorator
class Dataset(object):

    def __init__(self, n_splits, split_index):
        print("DATASET: You should only see this message once.")
        (self._trainImages, self._trainLabels), (self._testImages, self._testLabels) = tf.keras.datasets.fashion_mnist.load_data()

        # Cross validation
        skf = StratifiedKFold(n_splits=10)
        indices_by_expIdx = []
        for train_index, val_index in skf.split(self._trainImages, self._trainLabels):
            indices_by_expIdx.append((train_index, val_index))

        def convert_to_tf(data):
            # reshape data to fit shape
            data = data.astype('float32') / 255
            return np.expand_dims(data, axis=-1)

        def get_split(type, split_index):
            # Get the training or validation data+labels, by given split
            train, val = indices_by_expIdx[split_index]

            indices = train
            if type == 'validation':
                indices = val

            train_data = convert_to_tf(self._trainImages[indices])
            train_labels = tf.keras.utils.to_categorical(self._trainLabels[indices])
            return train_data, train_labels

        self.trainImages = lambda : get_split('train', split_index)[0]
        self.trainLabels = lambda : get_split('train', split_index)[1]

        self.valImages = lambda : get_split('validation', split_index)[0]
        self.valLabels = lambda : get_split('validation', split_index)[1]

        self.testImages = lambda : convert_to_tf(self._testImages)
        self.testLabels = lambda : tf.keras.utils.to_categorical(self._testLabels)

if __name__ == "__main__":
    dataset = Dataset();
