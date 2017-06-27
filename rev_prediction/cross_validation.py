from typing import Tuple

from sklearn import linear_model
import numpy as np

from rev_prediction.constants import LINEAR
from rev_prediction.utils import load_training_data, write_to_file, results_file

MODELS = {LINEAR: linear_model}

def get_folds(num_rows, chunk_size):
    rows_per_chunk = num_rows // chunk_size
    remaining_rows = num_rows % chunk_size
    folds = [(i * chunk_size, (i + 1) * chunk_size) for i in range(rows_per_chunk)]

    if remaining_rows:
        second_last_index = rows_per_chunk * chunk_size
        last_index = second_last_index + remaining_rows
        folds.append((second_last_index, last_index))

    return folds


def get_new_inputs_labels(X: np.ndarray, y: np.ndarray, val_set):
    """
        X: np.array of inputs
        y: np.array of labels
        val_set: tuple indexing rows of X and y that form the validation set

    """
    X_ret = [X[i] for i in range(X.shape[0]) if i < val_set[0] or i >= val_set[1]]
    y_ret = [y[i] for i in range(y.shape[0]) if i < val_set[0] or i >= val_set[1]]
    return np.array(X_ret), np.array(y_ret)


def cross_validate_model(X: np.ndarray, y: np.ndarray, chunk_size: int):
    num_rows = X.shape[0]
    if num_rows == 0:
        raise RuntimeError('Empty training matrix!')

    error_rates = []
    folds = get_folds(num_rows, chunk_size)
    for val_set in folds:
        print("current validation set = ", val_set)
        training_inputs, training_labels = get_new_inputs_labels(X, y, val_set)
        regression_model = linear_model.LinearRegression()
        regression_model.fit(training_inputs, training_labels)

        prediction = regression_model.predict(X[val_set[0]: val_set[1]])
        error_rates.append(np.mean((prediction - y[val_set[0]: val_set[1]]) ** 2))

    return error_rates

def get_trained_linear_model(X, y):
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    return reg


if __name__ == '__main__':
    essential_input_cols = (1, 2, 4, 5, 6)
    X, y = load_training_data(essential_input_cols)
    error_rates = cross_validate_model(X, y, chunk_size=1000)
    data = "input col used {}: Mean MSE: {} \n".format(essential_input_cols,
                                                       np.mean(np.array(error_rates)))
    write_to_file(results_file, [data], mode='a')
