import numpy as np


class NearestNeighbours():
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is an N * D numpy matrix; each row of X corresponds to one example.
            y is an 1-dimensional array of size N containing the labels.
        """
        self.Xtrain = X
        self.ytrain = y

    def predict(self, X):
        """ X is an N * D numpy matrix where each row is an example whose label
            we wish to predict

            Returns a 1-dimensional array of size N consisting of the labels corresponding
            to the examples in X
        """
        num_test = X.shape[0]

        y_pred = np.zeros(num_test, dtype=self.ytrain.dtype)

        for i in range(num_test):
            # get the row-wise sum of the absolute values of the diff of
            # the rows of self.Xtrain and the ith row of X
            distances = np.sum(np.abs(self.Xtrain - X[i, :]), axis=1)
            min_index = np.argmin(distances)
            y_pred[i] = self.ytrain[min_index]

        return y_pred
