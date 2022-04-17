import os
import json
import pandas as pd
import numpy as np

def preprocess_text(df, classes):

    y_return = []
    whitespaces = ['\n', '\t', '\r']
    preprocessed = []

    # drop empty values
    df.dropna(inplace=True)
    # encode data - convert string categories to numbers
    for y in df['tag']:
        y_return.append(classes[y.split('-')[-1]])

    df['tag'] = y_return

    # preprocess text
    for text in df['text']:
        # removing unwanted space
        text = text.lstrip().rstrip()

        # remove special chars
        for char in whitespaces:
            text.replace(char, "")

        preprocessed.append(text)

    df['text'] = preprocessed
    return df

if __name__ == "__main__":

    with open("configs.json", "r") as json_file:
        configs = json.load(json_file)

    data_path = configs["data_path"]

    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))
    df_valid = pd.read_csv(os.path.join(data_path, "valid.csv"))

    classes = configs["classes"]



    # preprocess text
    df_train_preprocessed = preprocess_text(df_train, classes)
    df_test_preprocessed = preprocess_text(df_test, classes)
    df_val_preprocessed = preprocess_text(df_valid, classes)


    print("Number of Train Samples: ", len(df_train_preprocessed))
    print("Number of Test Samples: ", len(df_test_preprocessed))
    print("Number of Validation Samples: ", len(df_val_preprocessed))

    # save preprocessed files
    df_train_preprocessed.to_csv(os.path.join(data_path, "train_preprocessed.csv"))
    df_test_preprocessed.to_csv(os.path.join(data_path, "test_preprocessed.csv"))
    df_val_preprocessed.to_csv(os.path.join(data_path, "valid_preprocessed.csv"))
