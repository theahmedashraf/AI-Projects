import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# Step 1: Prepare the ReducedMNIST Database
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) =  tf.keras.datasets.mnist.load_data()
train_mask = np.isin(y_train_orig, range(10))
test_mask = np.isin(y_test_orig, range(10))
X_train_full, y_train_full = x_train_orig[train_mask][:10000], y_train_orig[train_mask][:10000]
X_test, y_test = x_test_orig[test_mask][:2000], y_test_orig[test_mask][:2000]

# Normalize and reshape the data
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_train_full = np.expand_dims(X_train_full, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Step 2: Randomly Select X Samples
def random_sample(X, y, ratio):
    num_samples = int(len(X) * ratio)
    indices = np.random.choice(len(X), num_samples, replace=False)
    return X[indices], y[indices]

ratios = [0.2, 0.3, 0.4, 0.5]
X_samples = []
y_samples = []
X_synthetic = []
y_synthetic = []

for ratio in ratios:
    X_ratio, y_ratio = random_sample(X_train, y_train, ratio)
    X_samples.append(X_ratio)
    y_samples.append(y_ratio)

#ratios_synthetic = [0.8, 0.7, 0.6, 0.5]
for ratio in ratios_synthetic:
    X_ratio, y_ratio = random_sample(X_train, y_train, ratio)
    X_synthetic.append(X_ratio)
    y_synthetic.append(y_ratio)

# Step 3: Train GAN for Synthetic Data Generation
def build_generator():
    generator = Sequential()
    generator.add(Dense(128, input_dim=100, activation='relu'))
    generator.add(Dense(256, activation='relu'))
    generator.add(Dense(512, activation='relu'))
    generator.add(Dense(784, activation='sigmoid'))
    generator.add(Reshape((28, 28, 1)))
    return generator

def build_discriminator():
    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=(28, 28, 1)))
    discriminator.add(Dense(512, activation='relu'))
    discriminator.add(Dense(256, activation='relu'))
    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    return gan

generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Step 4: Generate Synthetic Data
def generate_synthetic_data(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, 100))
    synthetic_data = generator.predict(noise)
    return synthetic_data

# Step 5: Train Recognition Model
def build_lenet():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def train_recognition_model(X_train, y_train, X_val, y_val):
    lenet = build_lenet()
    lenet.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    lenet.fit(X_train, to_categorical(y_train), batch_size=128, epochs=1000, validation_data=(X_val, to_categorical(y_val)))
    return lenet

# MODEL 1 (Generate model with the full data)
reference_model = train_recognition_model(X_train_full, y_train_full, X_val, y_val)

# MODEL 2 (Generate the model again with only X samples)
# AND
# MODEL 3 (Generate the model again with X amount of real data and the rest from the synthetic data)
# Note: All the model has the same length as the reference model (only different combinations of real and synthetic data)

synthetic_models = []
reduced_models = []
len_reduced = len(X_samples)
len_synthetic = len(X_synthetic) # len_synthetic = len_ref - len_reduced

for i in range(len_reduced): 
    X_real = X_samples[i]
    y_real = y_samples[i]
    X_synthetic_temp = X_synthetic[i]
    y_synthetic_temp = y_synthetic[i]
    X_synthetic_temp2 = generate_synthetic_data(generator, len(X_synthetic_temp))
    X_combined = np.concatenate((X_real, X_synthetic_temp2))
    y_combined = np.concatenate((y_real, y_synthetic_temp))
    synthetic_model = train_recognition_model(X_combined, y_combined, X_val, y_val)
    synthetic_models.append(synthetic_model)
    reduced_model = train_recognition_model(X_real, y_real, X_val, y_val)
    reduced_models.append(reduced_model)

# Step 6: Evaluate Model Performance
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=0)
    return accuracy

# Evaluate Reference Model Performance
reference_accuracy = evaluate_model(reference_model, X_test, y_test)

# Evaluate Reduced and Combined Models Performance
reduced_accuracies = []
synthetic_accuracies = []
for model in reduced_models:
    accuracy = evaluate_model(model, X_test, y_test)
    reduced_accuracies.append(accuracy)

for model in synthetic_models:
    accuracy = evaluate_model(model, X_test, y_test)
    synthetic_accuracies.append(accuracy)

# Print the performance of the models
print("\n\n  Comparison between the three Models for each Ratio\n")
for i, ratio in enumerate(ratios):
    print("\t\t  Ratio = {}".format(ratio))
    print("\t   Model\t\tAccuracy")
    print("\tReference Model\t\t {:.4f}".format(reference_accuracy))
    print("\tReduced   Model\t\t {:.4f}".format(reduced_accuracies[i]))
    print("\tSynthetic Model\t\t {:.4f}".format(synthetic_accuracies[i]))
    print("\n\n")