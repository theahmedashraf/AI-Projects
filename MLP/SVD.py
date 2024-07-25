import tensorflow as tf
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
# SVD feature extraction
svd = TruncatedSVD(n_components=784)  # Adjust the number of components as needed
x_train_svd = svd.fit_transform(x_train_flat)
x_test_svd = svd.transform(x_test_flat)
# Normalize the SVD-transformed data by dividing by the maximum value
x_train_normalized = x_train_svd / np.max(x_train_svd)
x_test_normalized = x_test_svd / np.max(x_test_svd)
# Define the MLP model architecture using Keras Sequential API
def MLP(num_layers):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))  # Adjust the input shape based on the SVD output size
    for _ in range(num_layers-1):
        model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model
# Create and compile the model
num_layers = 3  # Adjust the number of layers (1, 3, or 5)
model = MLP(num_layers)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Start timing the training process
start_time = time.time()
# Train the model on the training data for 25 epochs, using a batch size of 50
history = model.fit(x_train_normalized, y_train_reduced, 
                    epochs=25, 
                    verbose=0, 
                    batch_size=50, 
                    validation_data=(x_test_normalized, y_test_reduced))
# End timing the training process
end_time = time.time()
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test_normalized, y_test_reduced)
# Print the test accuracy and processing time
if num_layers == 1:
    print('\n1 Hidden Layers:')
elif num_layers == 3:
    print('\n3 Hidden Layers:')
elif num_layers == 5:
    print('\n5 Hidden Layers:')
print('Test Accuracy:', test_accuracy)
print('Processing Time:', (end_time - start_time) * 1000, 'milliseconds\n\n')
