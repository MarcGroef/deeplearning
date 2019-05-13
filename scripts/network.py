#!/usr/bin/python
import tensorflow as tf

class Network:

    def __init__(self, experimentType):
        self.validExperimentTypes = ['control', 'batchnorm', 'dropout', 'l2', 'l1']
        assert(experimentType in self.validExperimentTypes), ("Invalid experiment type.. Please choose from:\n" + str(self.validExperimentTypes))
        self.experimentType = experimentType
        self.build()

    def build(self):
        regularizer = None
        if self.experimentType == 'l2':
            regularizer = tf.keras.regularizers.l2(0.001)
        elif self.experimentType == 'l1':
            regularizer = tf.keras.regularizers.l1(0.01)

        model = tf.keras.Sequential()
        # Must define the input shape in the first layer of the neural network
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1), kernel_regularizer=regularizer))
        if self.experimentType == "batchnorm":
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))


        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', kernel_regularizer=regularizer))
        if self.experimentType == "batchnorm":
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        if self.experimentType == 'dropout':
            model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer))

        if self.experimentType == 'dropout':

            model.add(tf.keras.layers.Dropout(0.5))
        if self.experimentType == "batchnorm":
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=regularizer))

        if self.experimentType == 'dropout':
            tf.keras.layers.Dropout(0.5)

        # Take a look at the model summary
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model


    def train(self, x_train, y_train, x_valid, y_valid, batchSize, epochs):
        return self.model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, validation_data=(x_valid, y_valid)).history

    def test(self, x_test, y_test):
        # Evaluate the model on test set
        score = model.evaluate(x_test, y_test, verbose=0)
        # Print test accuracy
        print('\n', 'Test accuracy:', score[1])

if __name__ == "__main__":
    net = Network('l1')
