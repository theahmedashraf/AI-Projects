import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import time
from scipy.fftpack import dct as DCT
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# Load the MNIST dataset
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()
# Reduce the dataset to 1000 examples per digit for training and 200 examples per digit for testing
train_mask = np.isin(y_train_orig, range(10))
test_mask = np.isin(y_test_orig, range(10))
x_train_reduced, y_train_reduced = x_train_orig[train_mask][:1000], y_train_orig[train_mask][:1000]
x_test_reduced, y_test_reduced = x_test_orig[test_mask][:200], y_test_orig[test_mask][:200]
# Reshape the data to flatten the images into 1D arrays
x_train_flat = x_train_reduced.reshape((x_train_reduced.shape[0], -1))
x_test_flat = x_test_reduced.reshape((x_test_reduced.shape[0], -1))
# DCT feature extraction: Apply 1D DCT along each row of the flattened images
x_train_dct = np.apply_along_axis(lambda x: DCT(x, norm='ortho'), 1, x_train_flat)
x_test_dct = np.apply_along_axis(lambda x: DCT(x, norm='ortho'), 1, x_test_flat)
# Normalize the DCT-transformed data by dividing by the maximum value
x_train_normalized = x_train_dct / np.max(x_train_dct)
x_test_normalized = x_test_dct / np.max(x_test_dct)
# Define the MLP model architecture using Keras Sequential API
def MLP(numLayers, numNeurons, numFeatures):
    model = Sequential()
    dropout_rate = 0.25
    if numLayers == 1:
        # 1 Hidden Layer
        model.add(Dense(numNeurons, activation='relu', input_shape=(numFeatures,)))
    elif numLayers == 3:
        # 3 Hidden Layers
        model.add(Dense(numNeurons, activation='relu', input_shape=(numFeatures,)))
        #model.add(Dropout(dropout_rate))
        '''
        The dropout rate is a hyperparameter in neural networks, particularly in models that use dropout regularization. 
        Dropout is a technique used to prevent overfitting in neural networks by randomly ignoring (or "dropping out") 
        a fraction of input units during training.
        For example, if each sample in the dataset has 784 features 
        (as in the case of the MNIST dataset where each image is 28x28 pixels), set numFeatures to 784.
        '''
        model.add(Dense(int(numNeurons/2), activation='relu'))
        model.add(Dense(int(numNeurons/4), activation='relu'))
    elif numLayers == 5:
        # 5 Hidden Layers
        model.add(Dense(numNeurons, activation='relu', input_shape=(numFeatures,)))
        model.add(Dense(int(numNeurons/1), activation='relu'))
        model.add(Dense(int(numNeurons/1), activation='relu'))
        model.add(Dense(int(numNeurons/1), activation='relu'))
        model.add(Dense(int(numNeurons/1), activation='relu'))
    # Output layer for classification
    model.add(Dense(10, activation='softmax')) # output classes = 10
    return model

numLayers = 5
# We start with a hidden layer of 512 neurons, then gradually decrease the number of neurons in subsequent layers. 
model = MLP(numLayers, numNeurons = 512, numFeatures = 784)
# Compile the model with Adam optimizer, categorical cross-entropy loss function, and accuracy metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Start timing the training process
start_time = time.time()
# Train the model on the training data for 20 epochs, using a batch size of 128
history = model.fit(x_train_normalized, tf.keras.utils.to_categorical(y_train_reduced),
                    batch_size=128,
                    epochs=20,
                    verbose = 0,
                    validation_data=(x_test_normalized, tf.keras.utils.to_categorical(y_test_reduced)))

# End timing the training process
end_time = time.time()
# Evaluate the model on the test data to get test loss and accuracy
test_loss, test_accuracy = model.evaluate(x_test_normalized, tf.keras.utils.to_categorical(y_test_reduced))
# Print the test accuracy and processing time
if numLayers == 1:
    print('\n1 Hidden Layers:')
elif numLayers == 3:
    print('\n3 Hidden Layers:')
elif numLayers == 5:
    print('\n5 Hidden Layers:')
print('Test Accuracy:', test_accuracy)
print('Processing Time:', (end_time - start_time) * 1000, 'milliseconds\n\n')