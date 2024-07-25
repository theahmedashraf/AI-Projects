import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import time
from sklearn.decomposition import PCA
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
# PCA feature extraction
pca = PCA(0.98)
pca = PCA(n_components=784)
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)
# Normalize the PCA-transformed data by dividing by the maximum value
x_train_normalized = x_train_pca / np.max(x_train_pca)
x_test_normalized = x_test_pca / np.max(x_test_pca)
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
        model.add(Dense(int(numNeurons/4), activation='relu'))
        model.add(Dense(int(numNeurons/8), activation='relu'))
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
# We start with a hidden layer of 512 neurons, then gradually decrease the number of neurons in subsequent layers. 
numLayers = 5
model_layer = MLP(numLayers, numNeurons = 512, numFeatures = 784)
# Compile the model with Adam optimizer, categorical cross-entropy loss function, and accuracy metric
model_layer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Start timing the training process
start_time = time.time()
# Train the model on the training data for 20 epochs, using a batch size of 128
history = model_layer.fit(x_train_normalized, tf.keras.utils.to_categorical(y_train_reduced),
                    batch_size=50,
                    epochs=20,
                    verbose = 0,
                    validation_data=(x_test_normalized, tf.keras.utils.to_categorical(y_test_reduced)))
# End timing the training process
end_time = time.time()
# Evaluate the model on the test data to get test loss and accuracy
test_loss, test_accuracy = model_layer.evaluate(x_test_normalized, tf.keras.utils.to_categorical(y_test_reduced))
# Print the test accuracy and processing time
if numLayers == 1:
    print('\n1 Hidden Layers:')
elif numLayers == 3:
    print('\n3 Hidden Layers:')
elif numLayers == 5:
    print('\n5 Hidden Layers:')
print('Test Accuracy:', test_accuracy)
print('Processing Time:', (end_time - start_time) * 1000, 'milliseconds\n\n')