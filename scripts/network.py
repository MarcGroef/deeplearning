import tensorflow as tf

class Network:
    
    def __init__(self):
        self.build()
        
    def build(self):
        model = tf.keras.Sequential()

        # Must define the input shape in the first layer of the neural network
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # Take a look at the model summary
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
    
    
    def train(self, x_train, y_train, x_valid, y_valid, batchSize, epochs):
        self.model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, validation_data=(x_valid, y_valid))
        
    def test(self, x_test, y_test):
        # Evaluate the model on test set
        score = model.evaluate(x_test, y_test, verbose=0)
        # Print test accuracy
        print('\n', 'Test accuracy:', score[1])
        
if __name__ == "__main__":
    net = Network()