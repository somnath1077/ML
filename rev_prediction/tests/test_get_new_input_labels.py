import numpy as np
from rev_prediction.cross_validation import get_new_inputs_labels


def test_get_new_input_labels():
    X = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5],
                  [4, 5, 6],
                  [5, 6, 7]])
    y = np.array([1, 2, 3, 4, 5])

    val_sets = [(0, 2), (3, 5)]

    sols = [(np.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]]), np.array([3, 4, 5])),
            (np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), np.array([1, 2, 3]))]

    for val_set, sol in zip(val_sets, sols):
        X_ret, y_ret = get_new_inputs_labels(X, y, val_set)
        assert np.array_equal(X_ret, sol[0])
        assert np.array_equal(y_ret, sol[1])
