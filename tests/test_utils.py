import numpy as np
import sys
sys.path.append("..")
from utils import *

def test_nearest_multiple():
    np.testing.assert_equal(713, nearest_multiple(720,31))

def test_add_to_stack():
    a, b = None, 5
    a = add_to_stack(a,b)
    assert a == [b]
    a = add_to_stack(a,b)
    assert a == [b,b]
