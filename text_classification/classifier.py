import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from text_classification.utils import load_training_data, error_rate, load_test_data, write_to_file, out_file


def train_model(inputs, labels, chunk_size):
    number_partitions = int(len(inputs) / chunk_size)
    folds = {(i * chunk_size, (i + 1) * chunk_size) for i in range(0, number_partitions)}
    err_rates = []

    classifier = None
    #vectorizer = CountVectorizer(min_df=1)
    vectorizer = TfidfVectorizer(min_df=1)
    for val_set in folds:
        classifier = MLPClassifier(solver='lbfgs',
                                   activation='logistic',
                                   alpha=1e-5,
                                   hidden_layer_sizes=(36, 2),
                                   random_state=1)


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


def predict(classifier, vectorizer, test_data):
    X_test = vectorizer.transform(test_data)
    pred = classifier.predict(X_test)
    return pred


if __name__ == '__main__':
    training_data, labels = load_training_data()
    test_data = load_test_data()

    chunk_size = len(test_data)
    classifier, vectorizer, mean_err, sd_err = train_model(training_data, labels, chunk_size)
    print("Mean error = ", mean_err)
    print("SD error = ", sd_err)

    pred = predict(classifier, vectorizer, test_data)
    write_to_file(out_file, pred)
