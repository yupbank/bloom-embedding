from bloom_embedding import bloom_embedding, reverse_embedding
import numpy as np
import scipy.sparse as sp
import pytest 

@pytest.fixture
def x():
    return np.where(np.random.random((10, 100))>0.5, 1, 0)
    return sp.random(10, 100)

def test_bloom_embedding(x):
    new_shape = 10
    _, y = bloom_embedding(x, m=new_shape)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == new_shape

def test_reverse_embedding(x):
    new_shape = 10
    mapping, y = bloom_embedding(x, m=new_shape)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == new_shape
    y = y.toarray()
    revered_x = reverse_embedding(mapping, y)
    assert revered_x.shape[0] == x.shape[0]
    assert revered_x.shape[1] == x.shape[1]



