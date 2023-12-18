from collections import Counter

from keras.src.optimizers.adam import Adam
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import Preprocessing
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense



file_path = 'train.xlsx'
df = pd.read_excel(file_path)

pre = Preprocessing('')
df['review_description'] = df['review_description'].apply(pre.preprocessing_methods)
# df['review_description'] = df['review_description']

train_data, test_data = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

# Tokenize and pad the text data
max_words = 10000  # Adjust based on your vocabulary size
max_len = 100  # Adjust based on the maximum length of your sequences

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['review_description'])

train_sequences = tokenizer.texts_to_sequences(train_data['review_description'])
train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')

test_sequences = tokenizer.texts_to_sequences(test_data['review_description'])
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

# Define and train the LSTM model:

embedding_dim = 16  # Adjust based on your choice
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')  # 3 classes: 1, 0, -1
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

label_mapping = {-1: 0, 0: 1, 1: 2}
# Convert sentiment values to numpy array
# Apply the mapping to your labels
train_labels_mapped = train_data['rating'].map(label_mapping).to_numpy()
test_labels_mapped = test_data['rating'].map(label_mapping).to_numpy()

# Train the model
model.fit(train_padded, train_labels_mapped, epochs=10, validation_data=(test_padded, test_labels_mapped))

loss, accuracy = model.evaluate(test_padded, test_labels_mapped)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")












# temp = Counter(" ".join(data).split()).most_common(200)
# df_word_frequencies = pd.DataFrame(temp, columns=['Word', 'Frequency'])

# print(df_word_frequencies.to_string(index=False))



