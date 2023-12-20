import keras.layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Dropout, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Embedding
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.src.legacy.preprocessing.text import Tokenizer
from preprocessing.preprocessing import Preprocessing
from keras.utils import Sequence


class CustomDataGenerator(Sequence):
    def __init__(self, data, labels, batch_size, isPredict=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        self.isPredict = isPredict

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        batch_data = self.data[start:end]
        if not self.isPredict:
            batch_labels = self.labels[start:end]

        # Pad the last batch if it's smaller than batch_size
        if len(batch_data) < self.batch_size:
            remaining_samples = self.batch_size - len(batch_data)
            padding_data = np.zeros((remaining_samples,) + batch_data.shape[1:])

            batch_data = np.concatenate([batch_data, padding_data])
            if not self.isPredict:
                padding_labels = np.zeros((remaining_samples,) + batch_labels.shape[1:])
                batch_labels = np.concatenate([batch_labels, padding_labels])
        if not self.isPredict:
            return batch_data, batch_labels
        else :
            return np.array(batch_data)



class TransformerModel:
    def __init__(self, train_data, test_data, max_words, max_len, epochs):
        self.max_words = max_words
        self.max_len = max_len
        self.train_data = train_data
        self.test_data = test_data

        # initialize Tokenizer for word embedding
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.train_data['review_description'])

        # hyperparameters
        self.ETA = 0.0001
        self.batch_size = 25

        self.compile_model(self.ETA, True)
        self.train_model(epochs)

    def compile_model(self, ETA, training):
        self.embedding_dim = 256  # Dimensionality of word embeddings
        self.num_attention_heads = 4  # Number of attention heads in the MultiHeadAttention layer
        self.ff_dim = 4  # Number of attention heads in the MultiHeadAttention layer
        self.num_blocks = 2  # Number of Transformer blocks
        self.dropout_rate = 0.1  # Dropout rate for regularization

        self.model = self.build_transformer(self.embedding_dim, self.num_attention_heads, self.ff_dim, self.num_blocks, self.dropout_rate, training)

        # Loss function
        loss_function = SparseCategoricalCrossentropy(from_logits=False)
        # Optimizer
        optimizer = Adam(learning_rate=ETA)
        # Metrics
        metrics = [SparseCategoricalAccuracy(name='accuracy')]
        # self.model.compile(optimizer=Adam(learning_rate=ETA), loss='sparse_categorical_crossentropy',
        #                    metrics=['accuracy'])
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    def initialize_input(self, embedding_dim):
        inputs = Input(shape=(self.max_len, embedding_dim), batch_size=self.batch_size)
        pos_encoding = self.positional_encoding(self.max_len, embedding_dim)

        if pos_encoding.shape[1] != self.max_len or pos_encoding.shape[2] != embedding_dim:
            raise ValueError("Mismatch in sequence length or embedding dimension in positional encoding.")

        x = keras.layers.Add()([inputs, pos_encoding])
        return x, inputs

    def build_transformer(self, embedding_dim, num_heads, ff_dim, num_blocks, dropout_rate, training):

        x, inputs = self.initialize_input(embedding_dim)
        for _ in range(num_blocks):
            # Multi-Head Self Attention
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)(x, x)

            # Layer Normalization after attention
            attention_output = LayerNormalization()(attention_output + x)

            # Dropout after attention
            attention_output = Dropout(dropout_rate, trainable=training)(attention_output)

            # Feedforward Neural Network
            ff_output = Dense(ff_dim, activation='relu')(attention_output)
            ff_output = Dense(self.embedding_dim)(ff_output)

            # Layer Normalization after feedforward
            x = LayerNormalization()(ff_output + attention_output)

            # Dropout after feedforward
            x = Dropout(dropout_rate, trainable=training)(x)

        # Global Average Pooling and Dense layer for classification
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(3, activation='softmax')(x)

        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model


    def get_angles(self, pos, k, d):
        # Get i from dimension span k
        i = k // 2
        # Calculate the angles using pos, i and d
        angles = pos / np.power(10000, 2 * i / d)

        return angles

    def positional_encoding(self, positions, d_model):
        # initialize a matrix angle_rads of all the angles
        angle_rads = self.get_angles(np.arange(positions)[:, np.newaxis],

        np.arange(d_model)[np.newaxis, :], d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        # END CODE HERE

        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = np.tile(pos_encoding, [self.batch_size, 1, 1])

        return tf.constant(pos_encoding, dtype=tf.float32)


    def train_model(self, epochs):
        label_mapping = {-1: 0, 0: 1, 1: 2}
        # Apply the mapping to your labels
        self.train_labels_mapped = self.train_data['rating'].map(label_mapping).to_numpy()
        self.train_padded = self.word_embedding(self.train_data['review_description'])

        data_generator = CustomDataGenerator(self.train_padded, self.train_labels_mapped, self.batch_size)
        self.model.fit(data_generator, epochs=epochs)

    def word_embedding(self, data):

        # Convert text to sequences of integers
        sequences = self.tokenizer.texts_to_sequences(data)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        embedding_layer = Embedding(input_dim=self.max_words, output_dim=self.embedding_dim)

        # Apply word embeddings to the padded sequences
        embedded_sequences = embedding_layer(padded_sequences)
        return embedded_sequences

    def get_best_batch_size(self, train_size):
        result = 1
        sq = int(train_size**0.5)
        for i in range(1, sq + 1):
            if train_size % i == 0:
                d1 = i
                d2 = train_size / i
                result = d1 if abs(d1 - sq) <= abs(d2 - sq) else d2

        return result

    def evaluate_model(self):
        label_mapping = {-1: 0, 0: 1, 1: 2}
        # Apply the mapping to your labels
        test_labels_mapped = self.test_data['rating'].map(label_mapping).to_numpy()
        test_padded = self.word_embedding(self.test_data['review_description'])

        print(test_padded.shape)
        self.batch_size = self.get_best_batch_size(test_padded.shape[0])

        self.compile_model(self.ETA, False)

        data_generator = CustomDataGenerator(test_padded, test_labels_mapped, self.batch_size)
        loss, accuracy = self.model.evaluate(data_generator)

        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        return accuracy, loss

    def predict_sentiment(self, new_data):
        # Preprocess the new data
        pre = Preprocessing()
        X = new_data['review_description'].apply(pre.preprocessing_methods)

        new_padded = self.word_embedding(X)
        self.batch_size = self.get_best_batch_size(new_padded.shape[0])
        self.compile_model(self.ETA, False)
        data_generator = CustomDataGenerator(new_padded, None, self.batch_size, True)

        # Make predictions using the trained model
        predictions = self.model.predict(data_generator)

        predicted_labels = predictions.argmax(axis=1)

        # Reverse the label mapping to get the original sentiment values
        reverse_label_mapping = {0: -1, 1: 0, 2: 1}
        original_sentiments = [reverse_label_mapping[label] for label in predicted_labels]

        result_df = pd.DataFrame({'ID': new_data['ID'], 'rating': original_sentiments})
        return result_df

