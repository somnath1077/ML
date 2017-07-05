from sklearn.metrics import classification_report


def report(predictions, y_test):
    return classification_report(y_true=y_test, y_pred=predictions)
