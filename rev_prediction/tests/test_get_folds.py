import pytest

from rev_prediction.cross_validation import get_folds


@pytest.mark.unit
def test_get_folds():
    n_rows = [5, 10, 17]
    chunk_sizes = [5, 12, 7]
    solutions = [[(0, 5)], [(0, 10)], [(0, 7), (7, 14), (14, 17)]]
    for row, chunk, sol in zip(n_rows, chunk_sizes, solutions):
        assert sol == get_folds(row, chunk)
