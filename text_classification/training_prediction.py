from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from text_classification.utils import load_training_data, load_test_data, write_to_file, out_file


def train_model_and_predict(inputs, labels, test_data, hidden_layer_size):
    classifier = MLPClassifier(solver='adam',
                               activation='logistic',
                               alpha=1e-5,
                               hidden_layer_sizes=hidden_layer_size,
                               random_state=1)

    vectorizer = CountVectorizer(min_df=1)
    X_train = vectorizer.fit_transform(inputs)
    y = labels

    print("started training ...")
    classifier.fit(X_train, y)
    print("finished training ...")

    X_validation = vectorizer.transform(test_data)
    pred = classifier.predict(X_validation)
    return pred


if __name__ == '__main__':
    inputs, labels = load_training_data()
    test_data = load_test_data()
    hidden_layer_size = (3, 3)
    pred = train_model_and_predict(inputs, labels, test_data, hidden_layer_size)
    write_to_file(out_file, pred)
