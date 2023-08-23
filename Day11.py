# DataSets are collection of images, videos, audio and text. (DS)
# DataSets are of Two types:

# 1. Annotated/labeled => Every Sample has a label with itself.
# Use for training the ML algorithm
# Train Test Data Split
# Training => We provide Sample and their labels.
# Testing => We provide Sample and we expect label.


# 2. Annotated/labeled => Every Sample has a label with itself


# Keras Mnist has 70000 images of hand written digits 0-9
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# (train_data,train_lbl),(test_data,test_lbl)

# Neural Network
# Input => Neural Network => Output
# Image => Neural Network => 0-9
# 28x28=784 => Neural Network => 0-9
# Neural Network has weights which keep changing during the training.

# One Dense Layer will connect input layer and hidden layer
# Another Dense Layer will connect hidden layer and output layer
# Activation Layer is required between the two dense layers.

# Brain has Neurons
# A Neuron accumulates all inputs it receives.+
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

# Zj = i=0.Summation M-1 Wij Xi

(train_data,train_lbl),(test_data,test_lbl)=mnist.load_data()

print("Training Data before reshape:",train_data.shape)
print("Testing Data before reshape:",test_data.shape)

train_data=train_data.reshape(60000,784)
test_data=test_data.reshape(10000,784)

train_lbl=to_categorical(train_lbl,10)
test_lbl=to_categorical(test_lbl,10)
print("Training Data after reshape:",train_data.shape)
print("Testing Data after reshape:",test_data.shape)

# create Neural Network
model=Sequential([
    Dense(100,input_dim=784), # N-100 hidden layer neurons, M input
    Activation('sigmoid'),
    Dense(10),  # 10 K Output
    Activation('softmax')
])

print(test_lbl)
print(train_lbl)
print(model.summary())

# Z=np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])
# def sigmoid(z):
#     return 1/(1+np.exp(-z))
#
# g=sigmoid(Z)
# plt.plot(Z,g)
# plt.show()
