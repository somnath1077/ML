#!/usr/bin/python3

import os


dir = os.path.dirname(__file__)
input_file = os.path.join(dir, './data/xtrain_obfuscated.txt')
labels_file = os.path.join(dir, './data/ytrain.txt')
trainings_file = os.path.join(dir, './data/xtest_obfuscated.txt')
out_file = os.path.join(dir, './data/ytest.txt')

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

def load_training_data():
    sentences = fetch_data(input_file)
    novels = fetch_data(labels_file)

    assert len(sentences) == len(novels)
    # print(len(sentences))
    return sentences, novels

