import numpy as np
import time
from numba import njit
import logging


def load_data(file_name):
    '''
        Load data based on user input.

        Parameters
        ----------
        file_name : name of the dataset ('phase1' or 'cap360')

        Returns
        ------
        Adj_pm : Adjacency matrix 
        Mod_adj_pm : copy of Adj_pm that will be modified
    '''
    if (file_name == 'phase1'):
        NO_METERS = 108
        NO_POLES = 117
        data_file = np.loadtxt('phase1.txt', dtype=np.int32)
    elif (file_name == 'cap360'):
        NO_METERS = 7172
        NO_POLES = 6550
        data_file = np.loadtxt('cap360.txt', dtype=np.int32)

    Adj_pm = np.zeros((NO_METERS, NO_POLES)) # Adjacency matrix [meters x poles]
    Mod_adj_pm = np.zeros((NO_METERS, NO_POLES)) # Modifiable Adjacency matrix [meters x poles]
    cov_meters = np.zeros(NO_METERS)     # zero vector for covered meters (assigned 0 for uncovered meters)
    for x in data_file:
        x[0] -= 1 # Converting to 0-index
        x[1] -= 1 # Converting to 0-index
        Adj_pm[x[0]][x[1]] = 1
        Mod_adj_pm[x[0]][x[1]] = 1

    return Adj_pm, Mod_adj_pm, cov_meters



@njit 
def preprocessing_rows(a):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    sums = np.sum(a, axis = 1)
    for i in range(x):
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

def preprocessing_cols(a):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    sums = np.sum(a, axis = 1)
    max_sum = np.mean(sums)

    # Preprocessing for cap360 dataset, can be changed to phase1 by iterating over all columns 

    nzc = [int(x) for x in np.argwhere((sums >= (max_sum-1)) & (sums <= (max_sum + 1)))] 
    for i in range(min(80, len(nzc))):
        contains = np.argwhere(np.all(a[nzc[i], :] >= a, axis = 1))
        if (contains.shape[0] > 1):
            for j in contains:
                if j > nzc[i]:
                    contained[j] = True
    return contained

'''
# numba based rule 3 preprocessing columns commented out
@njit 
def preprocessing_cols(a, no_cols):
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

# numpy based rule 2 preprocessing rows commented out
def preprocessing_rows(a):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    sums = np.sum(a, axis = 1)
    min_sum = np.min(sums[np.argwhere(sums > 0)])
    nzc = [int(x) for x in np.argwhere(sums == min_sum)]
    for i in range(min(30, len(nzc))):
        contains = np.argwhere(np.all(a[nzc[i], :] <= a, axis = 1))
        if (contains.shape[0] > 1):
            for j in contains:
                if j > nzc[i]:
                    contained[j] = True
    return contained
'''


def greedy_run(file_name, pre_flag):
    Adj_pm, Mod_adj_pm, cov_meters = load_data(file_name)
    start_time = time.time()
    list_cov_poles = []   # list of covering poles
    scores = np.sum(Mod_adj_pm, axis = 0)
    while (np.any(cov_meters == 0)): # until all meters are covered
        if (pre_flag == 'Y'):
            #################################################
            # Rule 2 rows preprocessing
            contains = preprocessing_rows(Mod_adj_pm)
            scores -= np.sum(Mod_adj_pm[contains], axis = 0)
            Mod_adj_pm[contains] = 0
            cov_meters[contains] += 1

            # Rule 3 columns preprocessing
            contains = preprocessing_cols(Mod_adj_pm.T)
            Mod_adj_pm[:, contains] = 0
            scores[contains] = 0 

            # Rule 1 removing singleton rows
            singleton_rows = [int(x) for x in np.argwhere(np.sum(Mod_adj_pm, axis = 1) == 1)] # sum along all rows and see if it is 1
            if singleton_rows: # if there is atleast one singleton row
                for i in range(len(singleton_rows)):
                    if (np.sum(Mod_adj_pm[singleton_rows[i]]) == 1):
                        pole = int(np.argwhere(Mod_adj_pm[singleton_rows[i], :] == 1))
                        list_cov_poles.append(pole)        
                        x = Adj_pm[:, pole]
                        met_poles = [int(m) for m in np.argwhere(x == 1)] 
                        scores -= np.sum(Mod_adj_pm[met_poles], axis = 0)
                        cov_meters[met_poles] += 1
                        Mod_adj_pm[met_poles, :] = 0 

        #####################################################
        # Greedy Algorithm
        indices = np.argmax(scores) # Finds the best pole
        # add the best pole (indices)
        list_cov_poles.append(indices)
        # find the corresponding column in the matrix
        x = Adj_pm[:, indices]

        # find the meters covered by this pole and add to overall covered meters
        # and set the corresponding row of the meter to zero
        met_poles = [int(m) for m in np.argwhere(x == 1)] 
        cov_meters[met_poles] += 1
        scores -= np.sum(Mod_adj_pm[met_poles], axis = 0)
        Mod_adj_pm[met_poles, :] = 0

        #####################################################
        # Clean up
        if (file_name == 'cap360'):
            clm = 10
        else:
            clm = 1
        if(len(list_cov_poles)%clm == 0):
            clean_up_indices = []
            for i in range(0, len(list_cov_poles)): # for each already selected pole(sp)
                sp = list_cov_poles[i]
                meters_by_sp = [int(m) for m in np.argwhere(Adj_pm[:, sp] == 1)] # meters covered by sp
                if (np.any((cov_meters[meters_by_sp]-1) <= 0) == False):
                    clean_up_indices.append(i)
                    cov_meters[meters_by_sp] -= 1
            for j in sorted(clean_up_indices, reverse=True):
                del list_cov_poles[j]
        #print("no of poles:", len(list_cov_poles))
        #print("no of covered meters:", np.sum(cov_meters != 0))

    print("Running time:", (time.time() - start_time))
    print("Number of poles:",len(list_cov_poles))


def modified_greedy_run(file_name, pre_flag):
    Adj_pm, Mod_adj_pm, cov_meters = load_data(file_name)
    start_time = time.time()
    list_cov_poles = []     # list of covering poles
    while (np.any(cov_meters == 0)): # until all meters are covered
        if (pre_flag == 'Y'):
            #################################################
            # Rule 2 rows preprocessing
            contains = preprocessing_rows(Mod_adj_pm)
            Mod_adj_pm[contains] = 0
            cov_meters[contains] += 1

            # Rule 3 columns preprocessing
            contains = preprocessing_cols(Mod_adj_pm.T)
            Mod_adj_pm[:, contains] = 0

            # Rule 1 removing singleton rows
            singleton_rows = [int(x) for x in np.argwhere(np.sum(Mod_adj_pm, axis = 1) == 1)] # sum along all rows and see if it is 1
            if singleton_rows: # if there is atleast one singleton row
                for i in range(len(singleton_rows)):
                    if (np.sum(Mod_adj_pm[singleton_rows[i]]) == 1):
                        pole = int(np.argwhere(Mod_adj_pm[singleton_rows[i], :] == 1))
                        list_cov_poles.append(pole)        
                        x = Adj_pm[:, pole]
                        met_poles = [int(m) for m in np.argwhere(x == 1)] 
                        cov_meters[met_poles] += 1
                        Mod_adj_pm[met_poles, :] = 0 

        #####################################################
        # Modified Greedy Algorithm
        sum_rows = np.sum(Mod_adj_pm, axis = 1)
        min_poles_meter = np.min(sum_rows[np.argwhere(sum_rows > 0)])

        # Generalized Scoring below | can be changed based on the preference
        if (file_name == 'cap360'):
            hard_to_cover = np.where(sum_rows == min_poles_meter)
            # scores = [np.sum(Mod_adj_pm[hard_to_cover], axis=0) > 0] * np.sum(Mod_adj_pm, axis=0) # Score 1
            scores = [np.sum(Mod_adj_pm[hard_to_cover], axis=0)] * np.sum(Mod_adj_pm, axis=0) # Score 2
            # Score 3 below
        elif (file_name == 'phase1'):
            max_t = 2 # maximum k for t-hard to cover 
            scores = np.sum(Mod_adj_pm, axis = 0)
            for t in range(int(min_poles_meter), int(min_poles_meter)+max_t):
                t_hard_to_cover = np.where(np.sum(Mod_adj_pm, axis = 1) <= t)
                scores = scores * ((np.sum(Mod_adj_pm[t_hard_to_cover], axis = 0))**(1/(t-min_poles_meter+1)))

        indices = np.argmax(scores) # Finds the best pole
        # add the best pole (indices)
        list_cov_poles.append(indices)
        # find the corresponding column in the matrix
        x = Adj_pm[:, indices]
        # find the meters covered by this pole and add to overall covered meters
        # and set the corresponding row of the meter to zero
        met_poles = [int(m) for m in np.argwhere(x == 1)] 
        cov_meters[met_poles] += 1
        Mod_adj_pm[met_poles, :] = 0
        #####################################################
        # Clean up
        if (pre_flag == 'N' and file_name == 'cap360'):
            clm = 15 # clean up modulo
        elif(file_name == 'phase1'):
            clm = 1
        if(len(list_cov_poles) % clm == 0): # Clean up every 15 times
            clean_up_indices = []
            for i in range(0, len(list_cov_poles)): # for each already selected pole(sp)
                sp = list_cov_poles[i]
                meters_by_sp = [int(m) for m in np.argwhere(Adj_pm[:, sp] == 1)] # meters covered by sp
                if (np.any((cov_meters[meters_by_sp]-1) <= 0) == False):
                    clean_up_indices.append(i)
                    cov_meters[meters_by_sp] -= 1
            for j in sorted(clean_up_indices, reverse=True):
                del list_cov_poles[j]
        #print("no of poles:", len(list_cov_poles))
        #print("no of covered meters:", np.sum(cov_meters != 0))

    print("Running time:", (time.time() - start_time))
    print("Number of poles:",len(list_cov_poles))


if __name__ == '__main__':
    print("Enter dataset (phase1 | cap360):")
    file_name = input()
    print("Preprocessing (Y|N)")
    pre_flag = input()
    modified_greedy_run(file_name, pre_flag)
    #greedy_run(file_name, pre_flag)
