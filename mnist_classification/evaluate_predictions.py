from sklearn.metrics import classification_report, confusion_matrix


def report(predictions, y_test):
    return classification_report(y_true=y_test, y_pred=predictions)
