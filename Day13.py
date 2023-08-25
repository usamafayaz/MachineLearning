#https://github.com/usamafayaz/MachineLearning.git
# Accuracy and Speed
# 1. #
# Ways to Increase Accuracy:
# 1. Adding some hidden layers.
# 2. Adding number of neurons in hidden layers.
# 3. Adding more images per class.
# 4. Data Augmentation.
#       Add Salt and Pepper noise in 20k images.
#       Add 20k noisy images.
#       Add 20k scaling
#       Add 20k blue filter images
#       Add 20k rotated images

# 5. OverFitting => More Accuracy in Training but less Accuracy in Testing.
# Solution of OverFitting
#   1. Early Stop.
        # Stop Training at point where gap between training and testing is minimum and loss is also less.
#   2. Drop Out.
        # Stop some neurons for some time. Kick Out brilliant students for some time.

# 6. DataSet Imbalance
    # A-100 || B-20 || C-40
    # A is majority class and B is minority class
    # Solution of DataSet Imbalance:
        # 1. OverSampling => increase in Minority class.
        # 2. UnderSampling => SMOTE algorithm.

###################################################################

# 2. #
# Speed and Real Time Performance.

###################################################################

# Feature Extraction
# Neural Network always works on number.

# Features => Defining Characteristics through which we could identify difference between one object from another.

# Hand Crafted 2012-2013 || Humans took out the Features.
# 2013 Concolution Neural Network (CNN) is revolutionary thing !

# Concolution Neural Network (CNN)

# 128x128  ==>  Conv Net 64 ==> 126x126x64 ==> Max Pooling ==> 63x63x64 ==> Conv Net 128 ==> 61x61x128 ==> Max Pooling ==> 30x30x128 ==> Conv Net 128 ==> 28x28x128 ==> Max Pooling ==> 14x14x128


# Loss Function
# Difference between actual output y and output produced by Neural Network y^
# l = y-y^
# Weights are learnable parameters.

# When we back propagate the loss function the weights will be changed.
# We place the line in the optimal place so that most of the points are correct.

# Gradient Descent Algorithm
# To find the minimum error.
# Pnew =Pold+StepSize => 0.01
# if StepSize=0 then Pnew =Pold
# rate of change in x axis / rate of change in y axis = gradient => derivative.

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import random

def LossandAccuracy():
    # laod mnist dataset
    classes = 10
    inputSize = 784
    (trainData, trainLbl), (testData, testLbl) = mnist.load_data()
    # Convert to categorical labels.
    trainLbl = to_categorical(trainLbl, classes)
    testLbl = to_categorical(testLbl, classes)

    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
        MaxPool2D((2,2)),
        Conv2D(64,(3,3)),
        MaxPool2D((2,2)),
        Flatten(),

        Dense(100, input_dim=inputSize,activation='sigmoid'),  # N-100 hidden layer neurons, M input
        Dense(classes,activation='softmax'),  # 10 K Output
    ])

    # Provide basic settings for neural network.
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')

    # Provide training for neural network.
    history = model.fit(trainData, trainLbl, epochs=20, batch_size=80, verbose=1)

    # Provide testing on the neural network.
    result = model.evaluate(testData, testLbl)
    print(model.summary())

    # save the trained model in the file.
    model.save("model.keras")


LossandAccuracy()