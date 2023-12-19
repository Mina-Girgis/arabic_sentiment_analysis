import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from preprocessing.preprocessing import Preprocessing


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
        ETA = 0.0001
        self.batch_size = 32

        self.compile_model(ETA)
        self.train_model(epochs)
    def compile_model(self, ETA):
        embedding_dim = 256  # Dimensionality of word embeddings
        num_attention_heads = 4  # Number of attention heads in the MultiHeadAttention layer
        ff_dim = 4  # Number of attention heads in the MultiHeadAttention layer
        num_blocks = 2  # Number of Transformer blocks
        dropout_rate = 0.1  # Dropout rate for regularization

        self.model = self.build_transformer(embedding_dim, num_attention_heads, ff_dim, num_blocks, dropout_rate)

        # Loss function
        loss_function = SparseCategoricalCrossentropy(from_logits=False)
        # Optimizer
        optimizer = Adam(learning_rate=ETA)
        # Metrics
        metrics = [SparseCategoricalAccuracy(name='accuracy')]
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    def build_transformer(self, embedding_dim, num_heads, ff_dim, num_blocks, dropout_rate):
        inputs = Input(shape=(self.max_len, embedding_dim))
        pos_encoding = self.positional_encoding(self.max_len, embedding_dim)
        x = tf.keras.layers.Concatenate(axis=-1)([inputs, pos_encoding])

        for _ in range(num_blocks):
            # Multi-Head Self Attention
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)(x, x)
            # Layer Normalization after attention
            attention_output = LayerNormalization()(attention_output + x)
            # Dropout after attention
            attention_output = Dropout(dropout_rate)(attention_output)

            # Feedforward Neural Network
            ff_output = Dense(ff_dim, activation='relu')(attention_output)
            ff_output = Dense(embedding_dim)(ff_output)
            # Layer Normalization after feedforward
            x = LayerNormalization()(ff_output + attention_output)
            # Dropout after feedforward
            x = Dropout(dropout_rate)(x)

        # Global Average Pooling and Dense layer for classification
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(3, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model



    def get_angles(self, pos, k, d):
        # Get i from dimension span k
        i = k // 2
        # Calculate the angles using pos, i and d
        angles = pos / np.power(10000, 2 * i / d)
        # END CODE HERE

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

        return tf.cast(pos_encoding, dtype=tf.float32)


    def train_model(self, epochs):
        label_mapping = {-1: 0, 0: 1, 1: 2}
        # Apply the mapping to your labels
        self.train_labels_mapped = self.train_data['rating'].map(label_mapping).to_numpy()
        self.test_labels_mapped = self.test_data['rating'].map(label_mapping).to_numpy()

        train_padded = self.word_embedding(self.train_data['review_description'])
        self.test_padded = self.word_embedding(self.test_data['review_description'])

        self.model.fit(train_padded, self.train_labels_mapped, epochs=epochs, batch_size=self.batch_size)

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.test_data['review_description'], self.test_data['rating'])
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        return accuracy, loss

    def word_embedding(self, data):
        data_sequences = self.tokenizer.texts_to_sequences(data)
        data_padded = pad_sequences(data_sequences, maxlen=self.max_len, padding='post')

        return data_padded

