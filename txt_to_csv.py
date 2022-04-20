import json
import os
import pandas as pd

def txt_to_lst(f, type):

    """
    Usage Notes:
    
    The purpose of this function is to convert the contents of txt files and returns a list independent and dependent variables
    txt_to_lst takes two arguments
    - f (file object) : the txt file to be opened
    - type (str) : whether the corresponding file is train, test or validation 
    
    Returns
    - X: independent text tokens (list of str)
    - y: dependent text tags for every token (list of str)
    
    """
    print("Converting ", type, "from txt to lists ...")
    f.seek(0)
    X = []
    y = []
    for line in f:

        # check for empty lines and file start string

        if len(line) > 0 and line[0] != "\n" and not line.startswith("-DOCSTART"):
            chunks = line.split(" ")
            word = chunks[0]

            if len(word) == 0:
                continue

            label = chunks[-1][:-1]

            data = {"word": word, "label": label}

            X.append(word)
            y.append(label)

    print("Done!")
    return X, y

if __name__ == "__main__":

    with open("configs.json", "r") as json_file:
        configs = json.load(json_file)

    data_path = configs["data_path"]

    f_train = open(os.path.join(data_path, "train.txt"))
    f_test = open(os.path.join(data_path, "test.txt"))
    f_val = open(os.path.join(data_path, "valid.txt"))

    # opening text file, putting contents into a list
    X_train , y_train = txt_to_lst(f_train, "Train")
    X_test , y_test = txt_to_lst(f_test, "Test")
    X_val , y_val = txt_to_lst(f_val, "Validation")

    # Checking if number of samples in X and y are same
    assert len(X_train) == len(y_train), "Number of samples DO NOT match in X and Y training set"
    assert len(X_test) == len(y_test), "Number of samples DO NOT match in X and Y test set"
    assert len(X_val) == len(y_val), "Number of samples DO NOT match in X and Y validation set"


    print("Number of Train Samples: ", len(X_train))
    print("Number of Test Samples: ", len(X_test))
    print("Number of Validation Samples: ", len(X_val))


    # Converting lists to dataframe
    df_train = pd.DataFrame(list(zip(X_train, y_train)),
                        columns=['text', 'tag'])
    df_test = pd.DataFrame(list(zip(X_test, y_test)),
                        columns=['text', 'tag'])
    df_val = pd.DataFrame(list(zip(X_val, y_val)),
                        columns=['text', 'tag'])

    # saving files
    df_train.to_csv(os.path.join(data_path, "train.csv"))
    df_test.to_csv(os.path.join(data_path, "test.csv"))
    df_val.to_csv(os.path.join(data_path, "valid.csv"))
