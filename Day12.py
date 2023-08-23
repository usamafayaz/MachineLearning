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
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import random

def showRandomImages():
    (trainData, trainLbl), _ = mnist.load_data()

    random_indices = random.sample(range(len(trainData)), 9)
    print(random_indices)

    plt.figure(figsize=(9, 4))

    for i, index in enumerate(random_indices, 1):
        plt.subplot(3, 3, i)
        plt.imshow(trainData[index], cmap='gray')
        plt.title(f'Label: {trainLbl[index]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def LossandAccuracy():
    # laod mnist dataset
    classes = 10
    inputSize = 784
    (trainData, trainLbl), (testData, testLbl) = mnist.load_data()
    # Perform Flattening Operation
    trainData = trainData.reshape(60000, inputSize)
    testData = testData.reshape(10000, inputSize)
    # Convert to categorical labels.
    trainLbl = to_categorical(trainLbl, classes)
    testLbl = to_categorical(testLbl, classes)

    model = Sequential([
        Dense(100, input_dim=inputSize),  # N-100 hidden layer neurons, M input
        Activation('sigmoid'),
        Dense(classes),  # 10 K Output
        Activation('softmax')
    ])

    # Provide basic settings for neural network.
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')

    # Provide training for neural network.
    history = model.fit(trainData, trainLbl, epochs=20, batch_size=80, verbose=1)

    # Provide testing on the neural network.
    result = model.evaluate(testData, testLbl)
    print("Accuracy of test is ", result[1])

    # save the trained model in the file.
    model.save("model.keras")

    epochs_list = []
    loss_list = []
    accuracy_list = []

    # Loop through the history and append the epoch number and loss value to the lists
    for epoch, loss in enumerate(history.history['loss'], 1):
        epochs_list.append(epoch)
        loss_list.append(loss)

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, history.history['accuracy'])
    plt.title('Accuracy X epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

showRandomImages()
# LossandAccuracy()

# (trainData, trainLbl), (testData, testLbl) = mnist.load_data()
#
# cv.imwrite('testImage.png', testData[0])
