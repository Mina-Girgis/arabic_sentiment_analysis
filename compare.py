from collections import Counter
from keras.src.optimizers.adam import Adam
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from Transformer import TransformerModel
from Transformer2 import TransformerModel2
from preprocessing.preprocessing import Preprocessing
import pandas as pd
import tensorflow as tf
import keras
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

def load_file(file_path):
    df = pd.DataFrame()
    if file_path.endswith("csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df

actual_output = load_file("test__label.csv")
predicted_output = load_file("validation2.csv")

s1 = actual_output['rating']
s2 = predicted_output['rating']

cnt_TP = 0
for x1,x2 in zip(s1,s2):
    if x1 == x2:
        cnt_TP = cnt_TP+1

print(cnt_TP/len(s1))