#!/usr/bin/python3

import os
import numpy as np

from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

dir = os.path.dirname(__file__)
input_file = os.path.join(dir, './xtrain_obfuscated.txt')
labels_file = os.path.join(dir, './ytrain.txt')
trainings_file = os.path.join(dir, './xtest_obfuscated.txt')
out_file = os.path.join(dir, './ytest.txt')

def error_rate():
    training_labels = fetch_data(labels_file)
    predicted_labels = fetch_data(out_file)
    err = [(i, j) for i, j in zip(training_labels, predicted_labels) if i != j]
    err_rate = (len(err) / len(training_labels)) * 100
    return err_rate

def write_to_file(file, data):
    with open(file, 'wb') as f:
        for datum in data:
            f.write(datum)

def fetch_data(file):
    lst = []
    with open(file, 'rb') as f:
        for line in f:
            lst.append(line)
    return lst

def vectorize(sentences: List[str]):
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(sentences)
    return X

def classify(X, y):
    classifier = MLPClassifier(solver='lbfgs',
                               activation='logistic',
                               alpha=1e-5,
                               hidden_layer_sizes = (8, 2),
                               random_state = 1)
    classifier.fit(X, y)
    return classifier

def predict(classifier, X):
    return classifier.predict(X)

def load_training_data():
    sentences = fetch_data(input_file)
    novels = fetch_data(labels_file)

    assert len(sentences) == len(novels)
    # print(len(sentences))
    return sentences, novels

if __name__ == '__main__':
    sentences, novels = load_training_data()
    X = vectorize(sentences)
    classifier = classify(X, novels)
    test_sentences = fetch_data(trainings_file)
    X_1 = vectorize(test_sentences)
    print("length of test_sentences = ", len(test_sentences))
    predictions = predict(classifier, X)
    write_to_file(out_file, predictions)
    print("Error rate = ", error_rate())