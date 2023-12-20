from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import nlpaug.augmenter.word as naw

from preprocessing.preprocessing import Preprocessing
from Transformer import TransformerModel
from LSTM import LSTM_model


def load_file(file_path):
    df = pd.DataFrame()
    if file_path.endswith("csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df


def save_file(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"{file_path} is saved completely")


def augment_text_with_synonyms(text):
    aug = naw.SynonymAug()
    augmented_text = aug.augment(text)
    return augmented_text[0]


if __name__ == "__main__":
    df = load_file('train.xlsx')

    # split the data
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

    # Apply Text Augmentation to Training Data for Class '0'
    class_0_samples = train_data[train_data['rating'] == 0].copy()
    class_0_samples['review_description'] = class_0_samples['review_description'].apply(augment_text_with_synonyms)
    train_data = pd.concat([train_data, class_0_samples], ignore_index=True)
    train_data = shuffle(train_data).reset_index(drop=True)

    pre = Preprocessing()
    train_data['review_description'] = train_data['review_description'].apply(pre.preprocessing_methods)
    test_data['review_description'] = test_data['review_description'].apply(pre.preprocessing_methods)

    # region Transformer Model
    lstm_model = LSTM_model(train_data, test_data, 10000, 100, 12)
    accuracy, loss = lstm_model.evaluate_model()

    validation_data = load_file("test _no_label.csv")
    predictions = lstm_model.predict_sentiment(validation_data)

    save_file(predictions, "validation_LSTM.csv")
    # endregion

    # region Transformer Model
    transformer_model = TransformerModel(train_data, test_data, 10000, 50, 12)
    accuracy, loss = transformer_model.evaluate_model()

    validation_data = load_file("test _no_label.csv")
    predictions = transformer_model.predict_sentiment(validation_data)

    save_file(predictions, "validation_Transformer.csv")
    # endregion
