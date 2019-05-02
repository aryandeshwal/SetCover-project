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
    for i in range(x):
        print(i)
        if (np.sum(a[i]) == 0): continue
        for j in range(i+1, x):
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
    for i in range(x):
        print(i)
        contains = np.argwhere(np.all(a[i, :] >= a, axis = 1))
        if (contains.shape[0] > 1):
            for j in contains:
                if j > i:
                    contained[j] = True
    return contained

def greedy_run(file_name, pre_flag):
    Adj_pm, Mod_adj_pm, cov_meters = load_data(file_name)
    start_time = time.time()
    list_cov_poles = []     # list of covering poles
    while (np.any(cov_meters == 0)): # until all meters are covered
        if (pre_flag == 'Y'):
            # Preprocessing (nps(no of preprocessing steps) times):    
            # 1. Removing singleton rows
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
                        print("Pole", pole, " added covering singleton row", singleton_rows[i])

            # 2. Removing rows that contain row j
            contains = preprocessing_rows(Mod_adj_pm)
            Mod_adj_pm[contains] = 0
            cov_meters[contains] += 1

            # 3. Removing columns contained in column j
            contains = preprocessing_cols(Mod_adj_pm.T)
            Mod_adj_pm[:, contains] = 0

        #####################################################
        # Greedy Algorithm
        indices = np.argmax(np.sum(Mod_adj_pm, axis = 0)) # Finds the best pole
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
        clean_up_indices = []
        for i in range(0, len(list_cov_poles)): # for each already selected pole(sp)
            sp = list_cov_poles[i]
            meters_by_sp = [int(m) for m in np.argwhere(Adj_pm[:, sp] == 1)] # meters covered by sp
            if (np.any((cov_meters[meters_by_sp]-1) <= 0) == False):
                clean_up_indices.append(i)
                cov_meters[meters_by_sp] -= 1
        for j in sorted(clean_up_indices, reverse=True):
            del list_cov_poles[j]
        print("no of poles:", len(list_cov_poles))
        print("no of covered meters:", np.sum(cov_meters != 0))

    print("Running time:", (time.time() - start_time))
    print("Number of poles:",len(list_cov_poles))

def modified_greedy_run(file_name, pre_flag):
    Adj_pm, Mod_adj_pm, cov_meters = load_data(file_name)
    start_time = time.time()
    list_cov_poles = []     # list of covering poles
    while (np.any(cov_meters == 0)): # until all meters are covered
        if (pre_flag == 'Y'):
            # Preprocessing (nps(no of preprocessing steps) times):    
            # 1. Removing singleton rows
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
                        print("Pole", pole, " added covering singleton row", singleton_rows[i])

            # 2. Removing rows that contain row j
            contains = preprocessing_rows(Mod_adj_pm)
            Mod_adj_pm[contains] = 0
            cov_meters[contains] += 1

            # 3. Removing columns contained in column j
            contains = preprocessing_cols(Mod_adj_pm.T)
            Mod_adj_pm[:, contains] = 0

        #####################################################
        # Modified Greedy Algorithm
        sum_rows = np.sum(Mod_adj_pm, axis = 1)
        min_poles_meter = np.min(sum_rows[np.nonzero(sum_rows)])
        print(min_poles_meter)
        hard_to_cover = np.where(np.sum(Mod_adj_pm, axis = 1) == min_poles_meter)
        scores = [np.sum(Mod_adj_pm[hard_to_cover], axis=0)] * np.sum(Mod_adj_pm, axis=0)
        print(scores)
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
        clean_up_indices = []
        for i in range(0, len(list_cov_poles)): # for each already selected pole(sp)
            sp = list_cov_poles[i]
            meters_by_sp = [int(m) for m in np.argwhere(Adj_pm[:, sp] == 1)] # meters covered by sp
            if (np.any((cov_meters[meters_by_sp]-1) <= 0) == False):
                clean_up_indices.append(i)
                cov_meters[meters_by_sp] -= 1
        for j in sorted(clean_up_indices, reverse=True):
            del list_cov_poles[j]
        print("no of poles:", len(list_cov_poles))
        print("no of covered meters:", np.sum(cov_meters != 0))

    print("Running time:", (time.time() - start_time))
    print("Number of poles:",len(list_cov_poles))


if __name__ == '__main__':
    print("Enter dataset:")
    #file_name = input()
    print("Preprocessing (Y|N)")
    #pre_flag = input()
    #modified_greedy_run(file_name, pre_flag)

    modified_greedy_run('phase1', 'N')
