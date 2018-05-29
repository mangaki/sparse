import numpy as np
from scipy.sparse import coo_matrix, find


Xi = np.array([
    [0, 3, 5],
    [1, 4, 5]
])
Xv = np.array([
    [1, 1, 1],
    [1, 1, 5]
])
y = [1, 0]


def lol_to_csr(Xi, Xv, y):
    nb_samples, nb_fields = Xi.shape
    rows = np.repeat(np.arange(nb_samples), nb_fields)
    cols = Xi.flatten()
    data = Xv.flatten()
    return coo_matrix((data, (rows, cols))).tocsr()

def csr_to_lol(X):
    nb_rows, nb_cols = X.shape
    rows, cols, values = find(X)
    Xi = [[] for _ in range(nb_rows)]
    Xv = [[] for _ in range(nb_rows)]
    for i, j, v in zip(rows, cols, values):
        Xi[i].append(j)
        Xv[i].append(v)
    return Xi, Xv


if __name__ == '__main__':
    X = lol_to_csr(Xi, Xv, y)
    print(X.todense())
    print(csr_to_lol(X))
