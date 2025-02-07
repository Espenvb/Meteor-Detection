# Meteor Detection usng a CNN

This project was completed by me and [@freds126](https://github.com/freds126). This repository is an edited version of the original code, making it clearer and removing elements that were only needed for a university task.

In this project, we implement a CNN for the classification of craters on Mars. Two datasets were provided: one with 2,783 labeled images and one with 904 unlabeled images. In the labeled dataset, there is an imbalance between images with craters and those without.
## Training

1. To enlarge the dataset, new versions of the images were created by rotating, flipping, and shifting.
2. A model was trained on the labeled data.
3. This model is then used on the unlabeled data, performing unsupervised learning to find the most likely images containing no craters to address the class imbalance.

Steps 2 and 3 are repeated until the dataset is more balanced.

## Model

We use a standard CNN architecture with convolutional layers followed by max pooling layers. For regularization, a dropout layer is used. The output has a sigmoid activation function and shows the probability of the image containing a crater.

## Results

The model achieved an F1 score of 82.2% during testing. No hyperparameter optimization was used, which, if implemented, could improve the performance of the model.
