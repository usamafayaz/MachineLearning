from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense

max_items=10000
outputClasses=46

(train_data,train_lbl),(test_data,test_lbl)=imdb.load_data(num_words=max_items)
max_len=500  # Every sentence is of 500.
embedding_size=128
train_data=pad_sequences(train_data,max_len)
test_data=pad_sequences(test_data,max_len)

train_lbl=to_categorical(train_lbl)
test_lbl=to_categorical(test_lbl)

model=Sequential([
    Embedding(max_items,embedding_size,input_length=max_len),
    SimpleRNN(embedding_size),
    Dense(1,activation='softmax')

])

# Provide basic settings for neural network.

model.compile(optimizer='sgd', loss='binary_crossloss', metrics='accuracy')

# Provide training for neural network.
history = model.fit(train_data, train_lbl, epochs=20, batch_size=100, verbose=1)

# Provide testing on the neural network.
score = model.evaluate(test_data, test_lbl)

print("Testing Loss is ",score)
print(model.summary())