from typing import List

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer

from text_classification.utils import load_training_data, fetch_data, write_to_file, out_file, error_rate, \
    trainings_file


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

if __name__ == '__main__':
    sentences, novels = load_training_data()
    X = vectorize(sentences)
    classifier = classify(X, novels)
    test_sentences = fetch_data(trainings_file)
    X_1 = vectorize(test_sentences)
    predictions = predict(classifier, X)
    write_to_file(out_file, predictions)
    print("Error rate = ", error_rate())


