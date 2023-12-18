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

from LSTM import LSTM_model

def load_file(file_path):
    df = pd.DataFrame()
    if file_path.endswith("csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df
def save_file(df, file_path="Validation.csv"):
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    df = load_file('train.xlsx')

    pre = Preprocessing()
    df['review_description'] = df['review_description'].apply(pre.preprocessing_methods)

    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

    lstm_model = LSTM_model(train_data, test_data, 10000, 100)
    accuracy, loss = lstm_model.evaluate_model()

    validation_data = load_file("test _no_label.csv")
    predictions = lstm_model.predict_sentiment(validation_data)

    print(predictions)
    save_file(predictions)
