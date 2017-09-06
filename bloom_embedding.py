import scipy.sparse as sp
import numpy as np
from sklearn.utils import murmurhash3_32
from itertools import groupby
from operator import itemgetter

K = 10
# K number of hash function

SEED = None
# SEED: None or K seed for hash function

M = 10000
# M targe domain capacity


def build_mapping(old_dimension, new_dimension, seeds):
    mapping = np.zeros((old_dimension, len(seeds)), dtype=int)
    for d in range(old_dimension):
        for n, seed in enumerate(seeds):
            mapping[d][n] = murmurhash3_32(d, seed=seed)%new_dimension
    return mapping


def bloom_embedding(X, k=K, m=M, seeds=SEED):
    seeds = seeds or range(k)
    row_dim, col_dim = X.shape
    assert len(seeds) == k
    mapping = build_mapping(col_dim, m, seeds)
    rows = []
    cols = []
    for row, indexes in groupby(zip(*X.nonzero()), key=itemgetter(0)):
        ones = set()
        for _, col in indexes:
            ones.update(mapping[col])
        cols.extend(sorted(ones))
        rows.extend([row for i in ones])

    data = np.ones(len(rows))
    return mapping, sp.csr_matrix((data, (rows, cols)), shape=(row_dim, m), dtype=int)


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
