''' 
Code to demonstrate the time difference between pure numpy based solution versus numba optimized solution
'''

import numpy as np
import time
from numba import njit

@njit 
def numba_preprocessing_rows(a, no_rows):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    sums = np.sum(a, axis = 1)
    for i in range(no_rows):
        if (sums[i] == 0): continue
        for j in range(i+1, x):
            if (sums[j] < sums[i]): continue
            if i != j and not contained[j]:
                equal = True
                for k in range(y):
                    if a[i, k] > a[j, k]:
                        equal = False
                        break
                contained[j] = equal
    return contained


@njit 
def numba_preprocessing_cols(a, no_cols):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    sums = np.sum(a, axis = 1)
    for i in range(no_cols):
        if (sums[i] == 0): continue
        for j in range(i+1, x):
            if (sums[j] < sums[i]): continue
            if i != j and not contained[j]:
                equal = True
                for k in range(y):
                    if a[i, k] < a[j, k]:
                        equal = False
                        break
                contained[j] = equal
    return contained



def numpy_preprocessing_rows(a, no_rows):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    sums = np.sum(a, axis = 1)
    for i in range(no_rows):
        contains = np.argwhere(np.all(a[i, :] <= a, axis = 1))
        if (contains.shape[0] > 1):
            for j in contains:
                if j > i:
                    contained[j] = True
    return contained



def numpy_preprocessing_cols(a, no_cols):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    sums = np.sum(a, axis = 1)
    for i in range(no_cols):
        contains = np.argwhere(np.all(a[i, :] >= a, axis = 1))
        if (contains.shape[0] > 1):
            for j in contains:
                if j > i:
                    contained[j] = True
    return contained



if __name__ == '__main__':
    NO_METERS = 7172
    NO_POLES = 6550
    data_file = np.loadtxt('cap360.txt', dtype=np.int32)

    Adj_pm = np.zeros((NO_METERS, NO_POLES)) # Adjacency matrix [meters x poles]
    for x in data_file:
        x[0] -= 1 # Converting to 0-index
        x[1] -= 1 # Converting to 0-index
        Adj_pm[x[0]][x[1]] = 1

    rows_np_list = []
    rows_numba_list = []
    cols_np_list = []
    cols_numba_list = []
    iter_values = [100, 200, 400, 800, 1000]
    for i in iter_values:
    	st = time.time()
    	contains = numpy_preprocessing_rows(Adj_pm, i)
    	rows_np_list.append(time.time()-st)
    print(rows_np_list)
    for i in iter_values:
    	st = time.time()
    	contains = numba_preprocessing_rows(Adj_pm, i)
    	rows_numba_list.append(time.time()-st)
    print(rows_numba_list)
    for i in iter_values:
    	st = time.time()
    	contains = numpy_preprocessing_cols(Adj_pm.T, i)
    	cols_np_list.append(time.time()-st)
    print(cols_np_list)
    for i in iter_values:
    	st = time.time()
    	contains = numba_preprocessing_cols(Adj_pm.T, i)
    	cols_numba_list.append(time.time()-st)   
    print(cols_numba_list)
