from collections import Counter

import numpy as np
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.preprocessing import Preprocessing
import pandas as pd

file_path = 'train.xlsx'
df = pd.read_excel(file_path)
review_description_column = df['review_description']
pre = Preprocessing('')
# print(review_description_column.head())
# print("-------------------------------")

# Define batch size
batch_size = 10  # Change this to your desired batch size

# Batch the data into smaller groups
batches = [review_description_column[i:i+batch_size] for i in range(0, len(review_description_column), batch_size)]

# Process each batch
processed_data = []
for idx, batch in enumerate(batches):
    print(f"Batch {idx + 1}:")
    curr_data = batch.apply(pre.preprocessing_methods)    # Replace this with your desired processing or analysis on each batch
    print(curr_data.head())
    processed_data.append(curr_data)  # Append processed data to the list
    print("\n")

data = pd.concat(processed_data, ignore_index=True)
print(data.head())


# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

# Get the most common words
most_common_words = Counter(" ".join(data).split()).most_common(50)
words_set = [word for word, _ in most_common_words]

# Get the indices of most common words in the TF-IDF vocabulary
word_indices = [tfidf_vectorizer.vocabulary_.get(word) for word in words_set]

# Filter the TF-IDF matrix to extract columns for the most common words
filtered_tfidf_matrix = tfidf_matrix[:, word_indices]

# Create a DataFrame with TF-IDF values for the most common words
df_tf_idf = pd.DataFrame(filtered_tfidf_matrix.toarray(), columns=words_set)
df_tf_idf['rating'] = df['rating']


excel_filename = 'word_importance_tfidf.xlsx'
df_tf_idf.to_excel(excel_filename, index=False)
print(f"TF-IDF results have been saved to {excel_filename}")
