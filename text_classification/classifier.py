from typing import List
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer

from text_classification.utils import load_training_data, fetch_data, \
    write_to_file, out_file, error_rate, test_data_file, input_file


def train_model(inputs, labels, chunk_size):
    number_partitions = int(len(inputs) / chunk_size)
    slices = {(i * chunk_size, (i + 1) * chunk_size) for i in range(0, number_partitions)}
    print(slices)
    err_rates = []

    classifier = MLPClassifier(solver='lbfgs',
                               activation='logistic',
                               alpha=1e-5,
                               hidden_layer_sizes=(10,),
                               random_state=1)
    for val_set in slices:
        print(val_set[0], val_set[1])
        training_sets = slices.difference({val_set})
        for set in training_sets:
            X = vectorize(inputs[set[0] : set[1]])
            y = labels[set[0] : set[1]]
            classifier.fit(X, y)


        X_val = vectorize(inputs[val_set[0] : val_set[1]])
        print(X_val)
        pred = classifier.predict(X_val)
        actual_labels = labels[val_set[0] : val_set[1]]
        err_rates.append(error_rate(pred, actual_labels))

    err_rates = np.array(err_rates)
    mean_err = np.mean(err_rates)
    sd_err = np.std(err_rates)
    return classifier, mean_err, sd_err

def vectorize(sentences: List[str]):
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(sentences)
    return X

def classify(X, y):
    classifier = MLPClassifier(solver='lbfgs',
                               activation='logistic',
                               alpha=1e-5,
                               hidden_layer_sizes = (10, ),
                               random_state = 1)
    classifier.fit(X, y)
    return classifier


def predict(classifier, X):
    return classifier.predict(X)

if __name__ == '__main__':
    sentences, novels = load_training_data()
    classifier, mean_err, sd_err = train_model(sentences, novels, 3000)
    print("Mean error = ", mean_err)
    print("SD error = ", sd_err)
    # X = vectorize(sentences)
    # classifier = classify(X, novels)
    # test_sentences = fetch_data(test_data_file)
    # X_1 = vectorize(test_sentences)
    # predictions = predict(classifier, X)
    # write_to_file(out_file, predictions)
    # print("Error rate = ", error_rate(predictions, novels))


