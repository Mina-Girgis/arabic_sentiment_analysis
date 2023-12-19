import pandas as pd
from keras.src.optimizers.adam import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from preprocessing.preprocessing import Preprocessing

class LstmModel:
    def __init__(self, train, test, max_words, max_len, epochs):
        self.train_data = train
        self.test_data = test
        self.max_words = max_words
        self.max_len = max_len

        # initialize Tokenizer for word embedding
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.train_data['review_description'])

        self.compile_model(ETA=0.0001)
        self.train_model(epochs=epochs)

    def word_embedding(self, data):
        data_sequences = self.tokenizer.texts_to_sequences(data)
        data_padded = pad_sequences(data_sequences, maxlen=self.max_len, padding='post')

        return data_padded

    def compile_model(self, ETA):
        embedding_dim = 16  # Adjust based on your choice
        self.model = Sequential([
            Embedding(input_dim=self.max_words, output_dim=embedding_dim, input_length=self.max_len),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(3, activation='softmax')  # 3 classes: 1, 0, -1
        ])

        self.model.compile(optimizer=Adam(learning_rate=ETA), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs):
        label_mapping = {-1: 0, 0: 1, 1: 2}
        # Apply the mapping to your labels
        self.train_labels_mapped = self.train_data['rating'].map(label_mapping).to_numpy()
        self.test_labels_mapped = self.test_data['rating'].map(label_mapping).to_numpy()

        train_padded = self.word_embedding(self.train_data['review_description'])
        self.test_padded = self.word_embedding(self.test_data['review_description'])

        print(train_padded)
        # Train the model
        self.model.fit(train_padded, self.train_labels_mapped, epochs=epochs, validation_data=(self.test_padded, self.test_labels_mapped))

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.test_padded, self.test_labels_mapped)
        # print(f"Test Loss: {loss}")
        # print(f"Test Accuracy: {accuracy}")
        return accuracy, loss

    def predict_sentiment(self, new_data):
        # Preprocess the new data
        pre = Preprocessing()
        X = new_data['review_description'].apply(pre.preprocessing_methods)

        new_padded = self.word_embedding(X)
        # Make predictions using the trained model
        predictions = self.model.predict(new_padded)

        predicted_labels = predictions.argmax(axis=1)

        # Reverse the label mapping to get the original sentiment values
        reverse_label_mapping = {0: -1, 1: 0, 2: 1}
        original_sentiments = [reverse_label_mapping[label] for label in predicted_labels]

        result_df = pd.DataFrame({'ID': new_data['ID'], 'rating': original_sentiments})
        return result_df