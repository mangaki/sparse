import numpy as np
from scipy.sparse import coo_matrix, find, save_npz
import sys


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

def ll_to_csr(ll):
    y = []
    rows = []
    cols = []
    data = []
    for i_row, line in enumerate(ll.splitlines()):
        tokens = line.split()
        y.append(int(tokens[0]))
        line_cols, line_data = zip(*[map(int, token.split(':')) for token in tokens[1:]])
        # print(tokens)
        # print(line_cols, line_data)
        rows.extend([i_row] * len(line_cols))
        cols.extend(line_cols)
        data.extend(line_data)
    X = coo_matrix((data, (rows, cols))).tocsr()
    return X, np.array(y)

if __name__ == '__main__':
    import time
    X = lol_to_csr(Xi, Xv, y)
    # print(X.todense())
    # print(csr_to_lol(X))
    X, y = ll_to_csr('0 1:2 3:4\n1 5:6')
    print(X.todense(), y)
    print(type(X[0].indices[0]))
    print(type(y[0]))
    np.save('wow.npy', y)

    # sys.exit(0)

    start = time.time()
    with open('/Users/jilljenn/code/liblinear/TRAINDEV.dat') as f:
        ll_train = f.read()
    X_train, y_train = ll_to_csr(ll_train)
    print('finished train', time.time() - start)
    start = time.time()
    with open('/Users/jilljenn/code/liblinear/TEST.dat') as f:
        ll_train = f.read()
    X_test, y_test = ll_to_csr(ll_train)
    print('finished test', time.time() - start)
    save_npz('X_train_fr_en.npz', X_train)
    save_npz('X_test_fr_en.npz', X_test)
    np.save('y_train_fr_en.npy', y_train)
    np.save('y_test_fr_en.npy', y_test)
    import pickle
    with open('fr_en_bestgen.pickle', 'wb') as f:
        pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)
