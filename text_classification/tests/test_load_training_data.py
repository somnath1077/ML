from text_classification.utils import load_training_data


def test_load_training_data():
    inputs, labels = load_training_data()
    assert len(inputs) == len(labels)