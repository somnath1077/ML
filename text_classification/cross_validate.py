import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from text_classification.utils import load_training_data, error_rate, load_test_data, err_file, get_new_inputs_labels


def cross_validate(inputs, labels, chunk_size, hidden_layer_size):
    number_partitions = int(len(inputs) / chunk_size)
    folds = [(i * chunk_size, (i + 1) * chunk_size) for i in range(0, number_partitions)]
    err_rates = []

    vectorizer = CountVectorizer(min_df=1)
    # vectorizer = TfidfVectorizer(min_df=1)
    for val_set in folds:
        classifier = MLPClassifier(solver='adam',
                                   activation='logistic',
                                   alpha=1e-5,
                                   hidden_layer_sizes=hidden_layer_size,
                                   random_state=1,
                                   learning_rate='adaptive')

        print("current validation set = ", val_set)
        training_set, label_set = get_new_inputs_labels(inputs, labels, val_set)
        X_train = vectorizer.fit_transform(training_set)
        classifier.fit(X_train, label_set)

        X_validation = vectorizer.transform(inputs[val_set[0]: val_set[1]])
        pred = classifier.predict(X_validation)
        actual_labels = labels[val_set[0]: val_set[1]]
        err_rates.append(error_rate(pred, actual_labels))

    err_rates = np.array(err_rates)
    mean_err = np.mean(err_rates)
    sd_err = np.std(err_rates)
    return mean_err, sd_err



def cross_validate_input():
    training_data, labels = load_training_data()
    test_data = load_test_data()

    hidden_layer_size = [(i, i) for i in [90]]
    chunk_size = len(test_data)

    with open(err_file, 'w') as f:
        for hidden_layer in hidden_layer_size:
            print("processing for hidden layer parameter {}".format(hidden_layer))
            mean_err, sd_err = cross_validate(training_data, labels, chunk_size, hidden_layer)

            mean_err = round(mean_err, 2)
            sd_err = round(sd_err, 2)
            message = "hidden layer size: {}".format(hidden_layer) + \
                      " mean error = {}".format(mean_err) + " sd error = {}".format(sd_err)
            f.write(message)
            f.write("\n")

if __name__ == '__main__':
    cross_validate_input()