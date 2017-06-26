from rev_prediction.utils import load_test_data


def test_load_testing_data():
    X = load_test_data()
    assert X.shape[1] == 5