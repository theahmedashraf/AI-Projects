This repository contains two projects that demonstrate the application of machine learning techniques on the ReducedMNIST dataset. The first project employs a Generative Adversarial Network (GAN) to generate synthetic data, while the second project uses a Multilayer Perceptron (MLP) to classify digits.

Project 1: Generative Adversarial Network (GAN)
Description
The objective of this project is to use a GAN to generate synthetic data for the ReducedMNIST dataset. 
The steps involved are:
1. Data Selection: Randomly select X samples (starting from 20%, 30%, 40%, and 50%) of each digit.
2. Training the GAN: Train a GAN network to generate new samples of each digit using the selected X samples. This process will be repeated four times for the different X ratios.
3. Quality Testing Pipeline:
 - Design a pipeline to test the quality of the generated data. The pipeline includes:
 - Training a recognition model (using LeNet 5 from a previous assignment) with full data as a reference model.
 - Re-training the recognition model with only X samples of the training data.
 - Generating synthetic data with sufficient variations.
 - Training the recognition model again using both X amount of real data and synthetic data.
 - Comparing the performance of the models trained with different data combinations and X ratios.


Project 2: Multilayer Perceptron (MLP)
Description
This project uses a Multilayer Perceptron (MLP) to classify digits in the ReducedMNIST dataset. 
The steps involved are:
1. Data Description:
 - ReducedMNIST training: 1000 examples for each digit.
 - ReducedMNIST test: 200 examples for each digit.
2. Model Training:
 - Using an MLP with 1, 3, or 5 hidden layers.
 - Employing features such as DCT, PCA, SVD, and LDA.
 - Experiment with different hyper-parameters to optimize performance.

