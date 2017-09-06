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
    hash_functions = [lambda r: murmurhash3_32(r, seed=seed) % new_dimension for seed in seeds]
    mapping = np.zeros((old_dimension, len(seeds)))
    for d in range(old_dimension):
        for n, function in enumerate(hash_functions):
            mapping[d][n] = function(d)
    return mapping


def bloom_embedding(X, k=K, m=M, seeds=SEED):
    seeds = seeds or range(k)
    assert len(seeds) == k
    mapping = build_mapping(X.shape[1], m, seeds)
    indptr = [0]
    indices = []
    for row, cols in groupby(zip(*X.nonzero()), key=lambda r: r[0]):
        ones = set()
        for row, col in cols:
            ones.update(mapping[col])
        indices.extend(ones)
        indptr.append(len(indices))
    data = np.ones(len(indices))
    return mapping, sp.csr_matrix((data, indices, indptr), dtype=int)


def reverse_embedding(mapping, Q):
    log_Q = np.log(Q)
    row_dim, embeding_dim = Q.shape
    reverse_dim, k = mapping.shape

    pred = np.zeros((row_dim, reverse_dim))

    for row in range(row_dim):
        for i in range(reverse_dim):
            for j in range(k):
                pred[row, i] += - log_Q[row, mapping[i, j]]
    return pred
