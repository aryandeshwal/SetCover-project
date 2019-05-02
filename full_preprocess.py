import numpy as np
import time
from numba import njit

# CONSTANTS
NO_METERS_C = 108
NO_POLES_C = 117
#NO_METERS_C = 7172
#NO_POLES_C = 6550
# Preprocessing included greedy algorithm
data_file = np.loadtxt('phase1.txt', dtype=np.int32)
# Create a meters x poles matrix
Adj_pm = np.zeros((NO_METERS_C, NO_POLES_C))
# another modifiable matrix 
Mod_adj_pm = np.zeros((NO_METERS_C, NO_POLES_C))
for x in data_file:
    x[0] -= 1 # Converting to 0-index
    x[1] -= 1
    Adj_pm[x[0]][x[1]] = 1
    Mod_adj_pm[x[0]][x[1]] = 1

@njit 
def preprocessing_rows(a):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    for i in range(x):
        print(i)
        for j in range(x):
            if i != j and not contained[j]:
                equal = True
                for k in range(y):
                    if a[i, k] > a[j, k]:
                        equal = False
                        break
                contained[j] = equal
    return contained

@njit 
def preprocessing_cols(a):
    x, y = a.shape
    contained = np.zeros(x, dtype=np.bool_)
    for i in range(x):
        for j in range(x):
            if i != j and not contained[j]:
                equal = True
                for k in range(y):
                    if a[i, k] < a[j, k]:
                        equal = False
                        break
                contained[j] = equal
    return contained


start_time = time.time()
# Create an zero vector for covered meters (assigned 0 for uncovered meters)
cov_meters = np.zeros(NO_METERS_C)
# list of covering poles
list_cov_poles = []
while (np.any(cov_meters == 0)):
    #####################################################
    # Preprocessing (nps(no of preprocessing steps) times):    
    # 1. Removing singleton rows
    # sum along all rows and see if it is 1
    singleton_rows = [int(x) for x in np.argwhere(np.sum(Mod_adj_pm, axis = 1) == 1)] 
    # Add the poles corresponding to the first nps singleton row
    if singleton_rows: # if there is atleast one singleton row
        for i in range(min(len(singleton_rows), nps)):
            if (np.sum(Mod_adj_pm[singleton_rows[i]]) == 1):
                pole = int(np.argwhere(Mod_adj_pm[singleton_rows[i], :] == 1))
                list_cov_poles.append(pole)        
                x = Adj_pm[:, pole]
                met_poles = [int(m) for m in np.argwhere(x == 1)] 
                cov_meters[met_poles] += 1
                Mod_adj_pm[met_poles, :] = 0 
                print("Pole", pole, " added covering ", singleton_rows[i], "row")

    # 2. Removing rows that contain row j
    
    contains = preprocessing_rows(Mod_adj_pm)
    print(contains)
    Mod_adj_pm[contains] = 0
    cov_meters[contains] += 1
    print(np.sum(cov_meters != 0))

    '''
    # 3. Removing columns contained in column j
    contains = preprocessing_rows(Mod_adj_pm.T)
    #print(contains)
    Mod_adj_pm[:, contains] = 0
    #cov_meters[contains] += 1
    '''
    #####################################################
    # Greedy Algorithm
    indices = np.argmax(np.sum(Mod_adj_pm, axis = 0))
    
    # add the best pole (indices)
    list_cov_poles.append(indices)
    # find the corresponding column in the matrix
    x = Adj_pm[:, indices]
    # find the meters covered by this pole and add to overall covered meters
    # and set the corresponding row of the meter to zero
    met_poles = [int(m) for m in np.argwhere(x == 1)] # time in orders of 10-4
    cov_meters[met_poles] += 1
    Mod_adj_pm[met_poles, :] = 0
    # Clean up
    if (len(list_cov_poles)):
        clean_up_indices = []
        for i in range(0, len(list_cov_poles)): # for each already selected pole(sp)
            sp = list_cov_poles[i]
            meters_by_sp = [int(m) for m in np.argwhere(Adj_pm[:, sp] == 1)] # meters covered by sp
            if (np.any((cov_meters[meters_by_sp]-1) <= 0) == False):
                clean_up_indices.append(i)
                print("\nremoving ", sp)
                cov_meters[meters_by_sp] -= 1
        for j in sorted(clean_up_indices, reverse=True):
            del list_cov_poles[j]
    print("no of poles:", len(list_cov_poles))
    print("no of covered meters:", np.sum(cov_meters != 0))
print((time.time() - start_time))
print(len(list_cov_poles))
print(list_cov_poles)
