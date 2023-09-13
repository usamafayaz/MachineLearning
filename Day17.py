# Problem: Neural Network only take Number.

# Images are naturally Numbers so no issue with them.
# Text is not a number, so we have to do some text embedding.
# Preserve the meaning of words in sentence.
# if i train my NN on word 'good', it should automatically be trained on word 'awesome'.

# Solution 1
# TF-IDF (Term Frequency Inverse document frequency)
# i.e, the man is in the hut.

# Term Frequency TF(the) = 2/6=1/3=0.33
# Inverse document frequency ITF(the) = 6/2=3

# Document 1 => The man is on the table.
# Document 2 => The man is mortal, so we all are mortal.
# Document Frequency DF(The) = Term occurring in all document/ No. of documents = 3/2
# Inverse Document Frequency IDF(The) = No. of documents / Term occurring in all document = 2/3
# TF-IDF = TF x IDF

# Solution 2
# Bags of Words (BOWs)
# Consumes alot of storage if using for large vectors.
# uses one hot encoding.

# Solution 3
# N-Grams
# The man is brave
# (The,man),(man,is),(is,brave)
# Not preserving the sentence structure.

# Solution 4

# Glove and BERT
# pre-trained Word Embedding.

# if we give vectors to words based on some features, it will make some sense.
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Load the IMDb dataset
num_words = 10000  # Number of most frequent words to use
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Define the maximum sequence length and pad sequences
max_seq_len = 200
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_seq_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_seq_len)

# Load the pre-trained GloVe word embeddings
glove_file = 'glove.6B.100d.txt'
embedding_dim = 100
word_embedding_dict = {}
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embedding_dict[word] = coefs

# Create an Embedding layer with weights initialized from GloVe
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, idx in imdb.get_word_index().items():
    if idx < num_words and word in word_embedding_dict:
        embedding_matrix[idx] = word_embedding_dict[word]

# Build the sentiment analysis model with GRU and GloVe embeddings
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
model.add(GRU(128, return_sequences=True))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)

# Evaluate the model
y_pred = (model.predict(x_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')