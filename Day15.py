# Sequence Modeling Application.
# Many to One
# One to Many
# Many to Many

# Agenda for today
# 1. Recurrent Neural Network (RNN)  famous are LSTM and GRU
#           1. Long Short Term Memory (LSTM)
#           2. Gated Recurrent Unit (GRU) faster version of LSTM

# FCNN doesnt have a memory. Y^=f(x)
# RNN have a memory. Cell State Yt^=f(Xt,Ht-1) => Y2^=f(X2,H1)
# H is a function of current input and previous state.

# 1. Recurrent Neural Network (RNN)
# Back Propagation Through Time. (BPTT)

# Prooblems with RNN
# 1. Handle variable-length sequences
# from keras.preprocessing.sequences import pad-sequences
# pad the small sentences to make them equal to the limit of the longest sentence.
# 2. Track long term Dependencies.  i.e I was born in France,  now I live in Boston, I speak fluent __
# 3. Maintaining difference in order i.e The food was good/bad, not bad/good at all.

# String should be converted into a number using Text Embedding.
# Text Embedding =>  convert the text into a numerical vectors in such a way that they are useful.

# Weights^no. of cells
# Problems =>
# 1. Vanishing Gradient (when weights are less than 1 i.e 0.5)
# 2. Exploding Gradient (when weights are greater than 1 i.e 2)

# If Past memory is required RNN is used.

# Reuters has 11228 articles with 11228 with 46 different categories.
from keras.datasets import reuters
from keras.preprocessing.sequence import  pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense
max_items=1000
outputClasses=46

(train_data,train_lbl),(test_data,test_lbl)=reuters.load_data(num_words=max_items)
max_len=500  # Every sentence is of 500.
embedding_size=64 # if we increase it by 128, accuracy will be increased.
train_data=pad_sequences(train_data,max_len)
test_data=pad_sequences(test_data,max_len)

train_lbl=to_categorical(train_lbl)
test_lbl=to_categorical(test_lbl)

model=Sequential([
    Embedding(max_items,embedding_size,input_length=max_len),
    SimpleRNN(embedding_size),
    Dense(outputClasses,activation='softmax')
])

# Provide basic settings for neural network.
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')

# Provide training for neural network.
history = model.fit(train_data, train_lbl, epochs=20, batch_size=80, verbose=1)

# Provide testing on the neural network.
result = model.evaluate(test_data, test_lbl)


print(model.summary())