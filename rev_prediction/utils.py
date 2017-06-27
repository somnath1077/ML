#!/usr/bin/python3

import os
import numpy as np


dir = os.path.dirname(__file__)
training_file = os.path.join(dir, './data/train.csv')
testing_file = os.path.join(dir, './data/prediction.csv')

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

def load_training_data():
    # Headers in training data file
    # 0. Date, 1. Keyword_ID, 2. Ad_group_ID, 3. Campaign_ID, 4. Account_ID,
    # 5. Device_ID, 6. Match_type_ID, 7. Revenue, 8. Clicks, 9. Conversions
    essential_input_cols = (1, 2, 4, 5, 6)
    output_cols = (7, 8)
    all_essential_cols = essential_input_cols + output_cols

    data = np.genfromtxt(training_file,
                         delimiter=',',
                         skip_header=False,
                         usecols=all_essential_cols)

    essential_cols_offset = [i for i in range(len(all_essential_cols))]
    x_cols = [i for i in range(len(essential_input_cols))]
    y_cols = essential_cols_offset[-len(output_cols):]

    X = data[:, x_cols]
    y = generate_y(data[:, y_cols])

    x_cols = X.shape[1]
    C = np.append(X, y, axis=1)
    C = remove_nans(C)
    X = C[:, [0, x_cols - 1]]
    y = C[:, [x_cols]]

    assert np.all(np.isfinite(X)) == True
    assert np.all(np.isfinite(y)) == True

    return X, y

def load_test_data():
    # 0. Date, 1. Keyword_ID, 2. Ad_group_ID, 3. Campaign_ID, 4. Account_ID,
    # 5. Device_ID, 6. Match_type_ID
    X_test = np.genfromtxt(testing_file,
                         delimiter=',',
                         skip_header=False,
                         usecols=(1,2,4,5,6))
    return remove_nans(X_test)
