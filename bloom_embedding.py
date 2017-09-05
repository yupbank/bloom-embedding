from functools import partial
from itertools import groupby
import scipy.sparse as sp
import numpy as np
from sklearn.utils import murmurhash3_32

K = 10
# K number of hash function

SEED = None
# SEED: None or K seed for hash function

M = 10000
# M targe domain capacity

def build_mapping(old_dimension, new_dimension, seeds):
    hash_functions = [lambda r: murmurhash3_32(np.int32(r), seed=seed)%new_dimension for seed in seeds]
    mapping = []
    for d in range(old_dimension):
        _mapping = []
        mapping.append(_mapping)
        for function in hash_functions:
            _mapping.append(function(d))
    return np.array(mapping)

def bloom_embedding(X, k=K, m=M, seeds=SEED):
    seeds = seeds or range(k)
    assert len(seeds) == k
    mapping = build_mapping(X.shape[1], m, seeds)
    new_rows, new_cols = [], []
    for row, indices in groupby(zip(*X.nonzero()), key=lambda r: r[0]):
        for row, col in indices:
            new_rows.extend([row for i in range(k)])
            new_cols.extend(mapping[col])
    return mapping, sp.csr_matrix((np.ones(len(new_cols)), (new_rows, new_cols)))


def reverse_embedding(mapping, Q, reverse_dimension):
    Q = np.log(Q)
    pred = np.zeros((Q.shape[0], mapping.shape[0]))
    for row, indices in groupby(zip(*Q.nonzero()), key=lambda r: r[0]):
        for row, col in indices:
            v = Q[row, col]
            for reversed_col in np.where(mapping==col)[0]:
                pred[row, reversed_col] += v
    return pred 


