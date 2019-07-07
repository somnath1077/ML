import numpy as np
import pandas as pd

def train_perceptron(data_set: np.array,
                     labels: np.array,
                     learning_rate: float,
                     num_iter: int) -> np.array:
    labels = labels.reshape(-1, 1)
    assert data_set.shape[0] == labels.shape[0]

    # adjust for bias
    num_inputs = data_set.shape[1] + 1
    num_training_pts = data_set.shape[0]

    X = np.insert(data_set, [0], [1], axis=1)
    w = np.random.ranf(num_inputs)

    print('initial weights: {}'.format(w))
    for i in range(num_iter):
        for idx in range(num_training_pts):
            x = X[idx]
            prod = np.dot(w, x)

            if prod > 0:
                out = 1
            else:
                out = -1
            target = labels[idx][0]

            for j in range(num_inputs):
                # print('Target: {}'.format(target))
                # print('Output: {}'.format(out))
                # print('w[{}]: {}'.format(j, w[j]))
                w[j] = w[j] + learning_rate * (target - out) * X[idx][j]
                # print('After adjustment: {}'.format(w[j]))
        print('Iter: {}, weights: {}'.format(i, w))

        #learning_rate /= 2
    return w

if __name__ == '__main__':
    data_set = pd.DataFrame({'x1': [0, 0, 1, 1],
                             'x2': [0, 1, 0, 1],
                             'x3': [-1, -1, -1, 1]})
    data = data_set[['x1', 'x2']].values
    labels = data_set['x3'].values
    w = train_perceptron(data_set=data, labels=labels, learning_rate=0.01, num_iter=1000)
    print(w)
