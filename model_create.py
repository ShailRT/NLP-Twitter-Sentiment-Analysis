import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("emotions.csv")  # Replace with your file

# Tokenization
VOCAB_SIZE = 10000
MAX_LEN = 50
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=MAX_LEN)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emotion'])
y = to_categorical(y)  # One-hot for multi-class

NUM_CLASSES = y.shape[1]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

EMBED_DIM = 128

model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)


import pickle

model.save("Model.h5")

with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)