from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from text_classification.utils import load_training_data, error_rate


def train_model(inputs, labels, chunk_size):
    number_partitions = int(len(inputs) / chunk_size)
    folds = {(i * chunk_size, (i + 1) * chunk_size) for i in range(0, number_partitions)}
    err_rates = []

    classifier = MLPClassifier(solver='lbfgs',
                               activation='logistic',
                               alpha=1e-5,
                               hidden_layer_sizes=(40, 5),
                               random_state=1,
                               warm_start=True)
    vectorizer = CountVectorizer(min_df=1)

    for val_set in folds:
        print("current validation set = ", val_set)
        training_sets = folds.difference({val_set})

        for set in training_sets:
            X_train = vectorizer.fit_transform(inputs[set[0]: set[1]])
            y = labels[set[0]: set[1]]
            classifier.fit(X_train, y)

        X_validation = vectorizer.transform(inputs[val_set[0]: val_set[1]])
        pred = classifier.predict(X_validation)
        actual_labels = labels[val_set[0]: val_set[1]]
        err_rates.append(error_rate(pred, actual_labels))

    err_rates = np.array(err_rates)
    mean_err = np.mean(err_rates)
    sd_err = np.std(err_rates)
    return classifier, vectorizer, mean_err, sd_err


def vectorize(sentences: List[str]):
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(sentences)
    return X


def classify(X, y):
    classifier = MLPClassifier(solver='lbfgs',
                               activation='logistic',
                               alpha=1e-5,
                               hidden_layer_sizes=(10,),
                               random_state=1)
    classifier.fit(X, y)
    return classifier


def predict(classifier, X):
    return classifier.predict(X)


if __name__ == '__main__':
    sentences, novels = load_training_data()
    classifier, vectorizer, mean_err, sd_err = train_model(sentences, novels, 3000)
    print("Mean error = ", mean_err)
    print("SD error = ", sd_err)
