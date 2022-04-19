import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import json
import os

from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc
import seaborn as sn
import matplotlib.pyplot as plt
sentences = []
tags = []
def format_word2vec(df, type):
    global sentences
    global tags
    print("Processing ", type, " data...")

    if type == 'valid':
        df = df.append({"text": ".", "tag": 0}, ignore_index=True)

    X = df['text'].values
    Y = df['tag'].values

    sentence = []
    tag = []
    for x in X:
        if x == '.':
            sentence.append(x)
            sentences.append(sentence)
            sentence = []
            continue

        sentence.append(x)

    # for i in range(len(X)):
    #     if X[i] == '.':
    #         sentence.append(X[i])
    #         tag.append(Y[i])
    #
    #         sentences.append(sentence)
    #         tags.append(tag)
    #
    #         sentence = []
    #         tag = []
    #
    #         continue
    #     sentence.append(X[i])
    #     tag.append(Y[i])


def convert_word2vec(df, model):

    X = df['text'].values
    X_vectors = []
    for x in X:
      X_vectors.append(model.wv[x])

    return X_vectors


def train_model(X_train_vectors, y_train, X_test_vectors, y_test, save_path, clf_type):
    # create the model, train it, print scores

    if clf_type == "RF":
        clf = RandomForestClassifier(n_estimators=200)
        clf_name = "Random Forest"
        report_name = "RF_classification_report.csv"
        #roc_name = "RF_ROC.png"

    if clf_type == "NB":
        clf_name = "Naive Bayes"
        clf = GaussianNB()
        report_name = "NB_classification_report.csv"
        #roc_name = "NB_ROC.png"

    if clf_type == "GB":
        clf_name = "Gradient Boosting"
        clf = clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,
        max_depth=1, random_state=0)
        report_name = "GB_classification_report.csv"
        #roc_name = "XGB_ROC.png"

    print("Training Classifier ", clf_name, "...")
    clf.fit(X_train_vectors, y_train)

    print("Train Score:", round(clf.score(X_train_vectors, y_train), 4)*100)
    print("Test Score:", round(clf.score(X_test_vectors, y_test), 4)*100)

    # Predicting the Test set results
    y_pred = clf.predict(X_test_vectors)

    report = metrics.classification_report(y_test, y_pred,  digits=5,
                                        target_names=["O", 'PER', 'LOC', 'MISC', 'ORG'],
                                        output_dict=True)
    # report = metrics.classification_report(y_test, y_pred,  digits=5,
    #                                     target_names=['PER', 'LOC', 'MISC', 'ORG'],
    #                                     output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(save_path, report_name))

    return clf
    #print((y_test==0).sum(), (y_pred==0).sum())

def custom_output(vector, clf):
    preds = clf.predict(vector)
    return preds

if __name__ == '__main__':

    with open("configs.json", "r") as json_file:
        configs = json.load(json_file)

    data_path = configs["data_path"]
    results_path = configs["results_path"]
    viz_path = configs["viz_path"]

    df_train = pd.read_csv(os.path.join(data_path, "train_preprocessed.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test_preprocessed.csv"))
    df_valid = pd.read_csv(os.path.join(data_path, "valid_preprocessed.csv"))

    format_word2vec(df_train, "train")
    format_word2vec(df_test, "test")
    format_word2vec(df_valid, "valid")

    # tags_new = []
    # sentences_new = []
    #
    # for i in range(len(tags)):
    #     sentence = sentences[i]
    #     tag = tags[i]
    #     if len(sentence) != len(tag):
    #         print(sentence, tag)
    #     to_pop = []
    #     for j in range(len(tag)):
    #         if tag[j] == 0:
    #             to_pop.append(j)
    #
    #
    #     tags_new.append([tag[k] for k in range(len(tag)) if k not in to_pop])
    #     sentences_new.append([sentence[k] for k in range(len(sentence)) if k not in to_pop])


    print(len(sentences))
    # print(len(sentences_new), tags_new.count(0))
    custom_model = Word2Vec(sentences, min_count=1,vector_size=300,workers=4, sg=1)
    # custom_model = Word2Vec(sentences_new, min_count=1,vector_size=300,workers=4, sg=1)

    X_train_vec = convert_word2vec(df_train, custom_model)
    X_test_vec = convert_word2vec(df_test, custom_model)
    X_val_vec = convert_word2vec(df_valid, custom_model)

    y_train = df_train['tag'].values
    y_test = df_test['tag'].values
    y_val = df_valid['tag'].values
    # X_train_vec = convert_word2vec(df_train[df_train['tag']!=0], custom_model)
    # X_test_vec = convert_word2vec(df_test[df_test['tag']!=0], custom_model)
    # X_val_vec = convert_word2vec(df_valid[df_valid['tag']!=0], custom_model)
    #
    # y_train = df_train[df_train['tag']!=0]['tag'].values
    # y_test = df_test[df_test['tag']!=0]['tag'].values
    # y_val = df_valid[df_valid['tag']!=0]['tag'].values
    #
    clf_rf = train_model(X_train_vec, y_train, X_test_vec, y_test, results_path, "RF")
    # clf_nb = train_model(X_train_vec, y_train, X_test_vec, y_test, results_path, "NB")
    # clf_gb = train_model(X_train_vec, y_train, X_test_vec, y_test, results_path, "GB")

    custom_inputs = configs["custom_inputs"]
    custom_tokens = word_tokenize(custom_inputs[0])
    #vectorize input
    custom_vector = []
    for t in custom_tokens:
        custom_vector.append(custom_model.wv[t])

    predicted_outputs = custom_output(custom_vector, clf_rf)

    classes_reversed = configs["classes_reversed"]
    mapped_preds = {}
    for i in range(len(predicted_outputs)):
        mapped_preds[custom_tokens[i]] = classes_reversed[str(predicted_outputs[i])]

    print("---| TOKEN | ------ | TAG | ------ \n")
    for token, tag in mapped_preds.items():
        print("---|", token, "| ------ | ", tag, " | ------ \n")
