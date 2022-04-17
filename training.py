import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc
import seaborn as sn
import matplotlib.pyplot as plt
sentences = []

def format_word2vec(df, type):
    global sentences

    print("Processing ", type, " data...")

    if type == 'valid':
        df = df.append({"text": ".", "tag": "O"}, ignore_index=True)

    X = df['text'].values

    sentence = []
    for x in X:
        if x == '.':
            sentence.append(x)
            sentences.append(sentence)
            sentence = []
            continue

        sentence.append(x)


def convert_word2vec(df, model):

    X = df['text'].values
    X_vectors = []
    for x in X:
      X_vectors.append(model.wv[x])

    return X_vectors

def plot_roc_curve(y_test, y_pred, save_path):

    ''' Plot the ROC curve for the target labels and predictions'''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc= auc(fpr,tpr)

    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig(save_path)
    print("ROC Plot saved in ", save_path)

def train_model(X_train_vectors, y_train, X_test_vectors, y_test, save_path, viz_path, clf_type):
    # create the model, train it, print scores

    if clf_type == "RF":
        clf = RandomForestClassifier(n_estimators=200)
        clf_name = "Random Forest"
        report_name = "RF_classification_report.csv"
        roc_name = "RF_ROC.png"

    if clf_type == "NB":
        clf_name = "Naive Bayes"
        clf = GaussianNB()
        report_name = "NB_classification_report.csv"
        roc_name = "NB_ROC.png"

    if clf_type == "XGB":
        clf_name = "Gradient Boosting"
        clf = XGBClassifier()
        report_name = "XGB_classification_report.csv"
        roc_name = "XGB_ROC.png"

    print("Training Classifier ", clf_name, "...")
    clf.fit(X_train_vectors, y_train)

    print("Train Score:", round(clf.score(X_train_vectors, y_train), 4)*100)
    print("Test Score:", round(clf.score(X_test_vectors, y_test), 4)*100)

    # Predicting the Test set results
    y_pred = clf.predict(X_test_vectors)

    report = metrics.classification_report(y_test, y_pred,  digits=5,
                                        target_names=["O", 'PER', 'LOC', 'MISC', 'ORG'],
                                        output_dict=True)

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(save_path, report_name))

    plot_roc_curve(y_test, y_pred, os.path.join(viz_path, roc_name))
    print("Classification Report saved in ", save_path)

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

    print(len(sentences))
    custom_model = Word2Vec(sentences, min_count=1,vector_size=300,workers=4, sg=1)

    X_train_vec = convert_word2vec(df_train, custom_model)
    X_test_vec = convert_word2vec(df_test, custom_model)
    X_val_vec = convert_word2vec(df_valid, custom_model)

    y_train = df_train['tag'].values
    y_test = df_test['tag'].values
    y_val = df_valid['tag'].values
    #
    train_model(X_train_vec, y_train, X_test_vec, y_test, results_path, viz_path, "RF")
    train_model(X_train_vec, y_train, X_test_vec, y_test, results_path, viz_path, "NB")
    train_model(X_train_vec, y_train, X_test_vec, y_test, results_path, viz_path, "XGB")
