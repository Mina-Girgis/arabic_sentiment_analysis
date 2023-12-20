from sklearn.model_selection import train_test_split

from Transformer import TransformerModel
from Transformer2 import TransformerModel2
from preprocessing.preprocessing import Preprocessing
import pandas as pd

from LSTM import LstmModel
from Transformer import TransformerModel

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

    #region LSTM Model
    # lstm_model = LstmModel(train_data, test_data, 10000, 100, 10)
    # # accuracy, loss = lstm_model.evaluate_model()
    #
    # validation_data = load_file("test _no_label.csv")
    # predictions = lstm_model.predict_sentiment(validation_data)
    #
    # print(predictions)
    # save_file(predictions)

    # region Transformer Model
    transformer_model = TransformerModel2(train_data, test_data, 10000, 100, 20)
    accuracy, loss = transformer_model.evaluate_model()

    validation_data = load_file("test _no_label.csv")
    predictions = transformer_model.predict_sentiment(validation_data)

    print(predictions)
    save_file(predictions, "validation2.csv")
    #endregion

    #region Transformer Model
    # transformer_model = TransformerModel(train_data, test_data, 1000, 256, 5)
    # accuracy, loss = transformer_model.evaluate_model()
    #
    # validation_data = load_file("test _no_label.csv")
    # predictions = transformer_model.predict_sentiment(validation_data)
    #
    # print(predictions)
    # save_file(predictions, "validation2")
