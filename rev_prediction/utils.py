#!/usr/bin/python3

import os
import numpy as np


dir = os.path.dirname(__file__)
training_file = os.path.join(dir, './data/train.csv')
testing_file = os.path.join(dir, './data/prediction.csv')

def generate_y(y_prime):
    y = np.zeros((y_prime.shape[0], 1))
    y[:, 0] = y_prime[:, 0] / y_prime[:, 1]
    y[y == np.inf] = 0
    y[y == -np.inf] = 0
    y[y == np.nan] = 0
    return y

def load_training_data():
    # Headers in training data file
    # 0. Date, 1. Keyword_ID, 2. Ad_group_ID, 3. Campaign_ID, 4. Account_ID,
    # 5. Device_ID, 6. Match_type_ID, 7. Revenue, 8. Clicks, 9. Conversions
    data = np.genfromtxt(training_file,
                        delimiter=',',
                        skip_header=False,
                        usecols=(1,2,4,5,6,7,8))

    X = data[:, [0, 1, 2, 3, 4]]
    y = generate_y(data[:, [5, 6]])
    return X, y


