from bloom_embedding import bloom_embedding, reverse_embedding
import numpy as np
import scipy.sparse as sp
import pytest 
import scipy.io

@pytest.fixture(params=[np.random.random((10, 100)), 
                        sp.random(10, 100).tocsr(), 
                        sp.random(10, 100, density=0).tocsr()])
def x(request):
    return request.param

def test_bloom_embedding(x):
    new_shape = 5
    _, y = bloom_embedding(x, m=new_shape)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == new_shape

def test_reverse_embedding(x):
    new_shape = 5
    mapping, y = bloom_embedding(x, m=new_shape)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == new_shape
    y = y.toarray()
    revered_x = reverse_embedding(mapping, y)
    assert revered_x.shape[0] == x.shape[0]
    assert revered_x.shape[1] == x.shape[1]
