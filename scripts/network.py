#!/usr/bin/python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

PARAMETER_EXPERIMENTS = ['l1_scalar', 'l2_scalar', 'fc_dropout_rate']#e.g. dropout rate

class Network:

    def __init__(self, experimentType, **kwargs):
        self.validExperimentTypes = ['control', 'batchnorm', 'dropout', 'l2', 'l1']
        assert(experimentType in self.validExperimentTypes), ("Invalid experiment type.. Please choose from:\n" + str(self.validExperimentTypes))
        self.experimentType = experimentType

        l_scalars = 0.001
        self.l1_scalar = kwargs['l1_scalar'] if 'l1_scalar' in kwargs else l_scalars
        self.l2_scalar = kwargs['l2_scalar'] if 'l2_scalar' in kwargs else l_scalars
        self.fc_dropout_rate = kwargs['fc_dropout_rate'] if 'fc_dropout_rate' in kwargs else .5

        print(kwargs)
        self.build()

    def build(self):
        regularizer = None
        if self.experimentType == 'l2':
            regularizer = tf.keras.regularizers.l2(self.l2_scalar)
        elif self.experimentType == 'l1':
            regularizer = tf.keras.regularizers.l1(self.l1_scalar)

        model = tf.keras.Sequential()
        # Must define the input shape in the first layer of the neural network
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1), kernel_regularizer=regularizer))
        if self.experimentType == "batchnorm":
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=2))


        model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', kernel_regularizer=regularizer))
        if self.experimentType == "batchnorm":
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizer))

        if self.experimentType == 'dropout':

            model.add(Dropout(self.fc_dropout_rate))
        if self.experimentType == "batchnorm":
            model.add(BatchNormalization())
        model.add(Dense(10, activation='softmax', kernel_regularizer=regularizer))

        # Take a look at the model summary
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model


    def train(self, x_train, y_train, x_valid, y_valid, batchSize, epochs):
        return self.model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, validation_data=(x_valid, y_valid)).history

    def test(self, x_test, y_test):
        # Evaluate the model on test set
        score = self.model.evaluate(x_test, y_test, verbose=0)
        # Print test accuracy
        print score
        print('\n', 'Test accuracy:', score[1])
        return score[1]

if __name__ == "__main__":
    net = Network('l1')
