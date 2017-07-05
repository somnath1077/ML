from sklearn.neural_network import MLPClassifier


def train_model(inputs, labels, hidden_layer_size):
    classifier = MLPClassifier(solver='adam',
                               activation='logistic',
                               alpha=1e-5,
                               hidden_layer_sizes=hidden_layer_size,
                               random_state=1)
    print("Training MLP classifier with hidden layer config = {}".format(hidden_layer_size))
    classifier.fit(inputs, labels)
    print("Done ...")
    return classifier


def predict(classifier, test_data):
    print("Predicting test data ..")
    predictions = classifier.predict(test_data)
    print("Done ...")
    return predictions
