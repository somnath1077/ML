from text_classification.utils import get_new_inputs_labels


def test_get_new_inputs_labels():
    inputs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    segments = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    for seg in segments:
        new_inp, new_lab = get_new_inputs_labels(inputs, labels, seg)
        assert len(new_inp) == 8
        assert len(new_lab) == 8
        for i, j in zip(new_inp, new_lab):
            assert int(i) == j
        print(new_inp)
        print(new_lab)

