# LSTM (Long Short Term Memory)
# Activation functions
# 1. Sigmoid (For Controlling)
# 0 or full weight
# 2. Tanh (For Penalization)
# 0, good memory or bad memory

# LSTM has different gates.
# 1. Forget Gate: How much memory to dropped from long term memory./ How much memory to keep
# Long term memory is multiplied with activation function sigmoid.
# 2. Update Gate: Store wheather the memory is positive or negative.
# 3. Output Gate: controls what information is sent to the next time step.

# Sentimental Ananlysis using LSTM.

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["I love this movie!", "This is awful.", "Great film.", "Terrible experience."]
labels = [1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Tokenization and preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Create a simple LSTM model
model = Sequential([
Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_sequence_length),
LSTM(64),
Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(np.array(padded_sequences), np.array(labels), epochs=10, batch_size=2)

# Test the model
test_texts = ["I kinda love this movie!", "The movie was some kind of awful shit."]
test_sequences = tokenizer.texts_to_sequences(test_texts)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)
predictions = model.predict(padded_test_sequences)

for text, sentiment in zip(test_texts, predictions):
    sentiment_label = "Positive" if sentiment > 0.5 else "Negative"
    print(f"Text: {text} / Sentiment: {sentiment_label} / Confidence: {sentiment[0]}")