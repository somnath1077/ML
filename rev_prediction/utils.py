#!/usr/bin/python3

import os
import numpy as np
import sys

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

dir = os.path.dirname(__file__)
training_file = os.path.join(dir, './data/train.csv')
testing_file = os.path.join(dir, './data/prediction.csv')
results_file = os.path.join(dir, './results/results.txt')

def generate_y(y_prime):
    y = np.zeros((y_prime.shape[0], 1))
    y[:, 0] = y_prime[:, 0] / y_prime[:, 1]

    return y


def remove_nans(C):
    D = C[~np.isnan(C).any(axis=1)]
    D[D == np.inf] = 0
    D[D == -np.inf] = 0
    D[D == np.nan] = 0
    return D


def load_training_data(essential_input_cols=(1, 2, 4, 5, 6), output_cols=(7, 8), num_rows=20000):
    # Headers in training data file
    # 0. Date, 1. Keyword_ID, 2. Ad_group_ID, 3. Campaign_ID, 4. Account_ID,
    # 5. Device_ID, 6. Match_type_ID, 7. Revenue, 8. Clicks, 9. Conversions
    all_essential_cols = essential_input_cols + output_cols
    data = np.genfromtxt(training_file,
                         delimiter=',',
                         skip_header=False,
                         usecols=all_essential_cols)

    x_cols, y_cols = get_X_y_cols(essential_input_cols, output_cols)
    print("x_cols = ", x_cols)
    print("y_cols = ", y_cols)
    X, y = get_X_y_arrays(data, x_cols, y_cols)

    X_trans = transform_categorical_data(X)
    return X_trans, y


def transform_categorical_data(X):
    """
        X is a matrix of categorical features.
        OneHotEncoder is used to encode the categorical data into
    """
    enc = OneHotEncoder()
    label_encoder = LabelEncoder()

    print("The number of cols in X = ", X.shape[1])
    for col in range(X.shape[1]):
        X[:, col] = label_encoder.fit_transform(X[:, col])

    X_trans = enc.fit_transform(X)
    return X_trans


def get_X_y_arrays(data, x_cols, y_cols):
    X = data[:, x_cols]
    y = generate_y(data[:, y_cols])

    num_X_cols = X.shape[1]
    C = np.append(X, y, axis=1)
    C = remove_nans(C)
    X = C[:, [i for i in range(num_X_cols)]]
    y = C[:, [num_X_cols]]

    assert np.all(np.isfinite(X)) == True
    assert np.all(np.isfinite(y)) == True
    return X, y


def get_X_y_cols(essential_input_cols, output_cols):
    all_essential_cols = essential_input_cols + output_cols
    essential_cols_offset = [i for i in range(len(all_essential_cols))]
    x_cols = [i for i in range(len(essential_input_cols))]
    y_cols = essential_cols_offset[-len(output_cols):]
    return x_cols, y_cols


def load_test_data(essential_input_cols=(1, 2, 4, 5, 6)):
    # 0. Date, 1. Keyword_ID, 2. Ad_group_ID, 3. Campaign_ID, 4. Account_ID,
    # 5. Device_ID, 6. Match_type_ID

    X_test = np.genfromtxt(testing_file,
                           delimiter=',',
                           skip_header=False,
                           usecols=essential_input_cols)
    return remove_nans(X_test)


def write_to_file(file, data, mode='wb'):
    with open(file, mode) as f:
        for datum in data:
            f.write(datum)
