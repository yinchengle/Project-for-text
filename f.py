import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

text = """g
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, 
it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, 
it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, 
we were all going direct to Heaven, we were all going direct the other way – in short, the period was so far like the present period, 
that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.
"""
# Create character-to-index and index-to-character mappings
chars = sorted(set(text))
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}
vocab_size = len(chars)

# Create sequences of fixed length
window_size = 40  # Length of each input sequence
step = 3  # Step size for creating sequences
sequences = []
next_chars = []

for i in range(0, len(text) - window_size, step):
    sequences.append(text[i:i + window_size])
    next_chars.append(text[i + window_size])

# Convert characters to indices
X = np.array([[char_to_index[char] for char in seq] for seq in sequences])
y = np.array([char_to_index[char] for char in next_chars])

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=vocab_size)

# Split data into training and validation sets
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

# 2. Define the model
embedding_dim = 128
rnn_units = 256

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=window_size),
    tf.keras.layers.GRU(rnn_units, return_sequences=True),
    tf.keras.layers.GRU(rnn_units),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train the model
epochs = 10
batch_size = 32

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val))

    # 4. Generate text after each epoch
    start_index = np.random.randint(0, len(X_val) - 1)
    generated_text = "".join(index_to_char[idx] for idx in X_val[start_index])
    print(f"Seed text: {generated_text}")

    for _ in tqdm(range(100)):
        input_sequence = np.array([char_to_index[char] for char in generated_text[-window_size:]])
        input_sequence = np.expand_dims(input_sequence, axis=0)

        predicted_probabilities = model.predict(input_sequence, verbose=0)[0]
        predicted_index = np.argmax(predicted_probabilities)
        predicted_char = index_to_char[predicted_index]

        generated_text += predicted_char

    print(f"Generated text: {generated_text}")

# 5. Save the model
model.save("text_gen_model.h5")
print("Model saved as 'text_gen_model.h5'")
