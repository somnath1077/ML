from rev_prediction.utils import load_training_data


def test_load_training_data():
    X, y = load_training_data()
    assert X.shape[0] == y.shape[0]
