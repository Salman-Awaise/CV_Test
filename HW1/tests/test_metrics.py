import numpy as np
from evaluation.metrics import angular_error, compute_statistics

def test_angular_error():
    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])
    err = angular_error(v1, v2)
    assert np.isclose(err, 90)
    err2 = angular_error(v1, v1)
    assert np.isclose(err2, 0)

def test_compute_statistics():
    errors = np.array([1,2,3,4,5,10,20])
    stats = compute_statistics(errors)
    assert 'mean' in stats and 'median' in stats and 'perc95' in stats
    assert stats['pct_lt3'] > 0
