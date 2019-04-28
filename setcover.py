import numpy as np
import time
# CONSTANTS
NO_METERS_P = 108
NO_POLES_P = 117
NO_METERS_C = 7172
NO_POLES_C = 6550
data_file = np.loadtxt('cap360.txt', dtype=np.int32)
# Create a meters x poles matrix
Adj_pm = np.zeros((NO_METERS_C, NO_POLES_C))
# another modifiable matrix
Mod_adj_pm = np.zeros((NO_METERS_C, NO_POLES_C))
for x in data_file:
    if (x[1]-1 < 0):
        print("sime")
        print(x)
    x[0] -= 1 # Converting to 0-index
    x[1] -= 1
    Adj_pm[x[0]][x[1]] = 1
    Mod_adj_pm[x[0]][x[1]] = 1

start_time = time.time()
# Create an empty for covered meters
cov_meters = np.zeros(NO_METERS_C)
# list of covered meters
list_cov_meters = []
# list of covering poles
list_cov_poles = []
total = 0
while (np.any(cov_meters == 0)):
    # get sorting indices
    st = time.time()
    indices = np.argsort(np.sum(Mod_adj_pm, axis = 0))[::-1]
    total += (time.time()-st)
    # add the best pole (indices[0])  
    list_cov_poles.append(indices[0])
    #print(indices[0],"pole added")
    # find the corresponding column in the matrix
    x = Adj_pm[:, indices[0]]
    #j += 1
    # find the meters covered by this pole and add to overall covered meters
    # and set the corresponding row of the meter to zero
    met_poles = np.argwhere(x == 1) # orders of 10-4
    for met_idx in met_poles:
        cov_meters[int(met_idx)] += 1
        Mod_adj_pm[met_idx, :] = 0
    # Clean up
    #st = time.time()
    for i in range(0, len(list_cov_poles), 5): # for each already selected pole(sp)
        if (i >= len(list_cov_poles)):
            break
        sp = list_cov_poles[i]
        cleanable = True
        meters_by_sp = np.argwhere(Adj_pm[:, sp] == 1) # meters covered by sp
        #print(meters_by_sp)
        for x in (meters_by_sp):
            if ((cov_meters[int(x)] - 1) == 0):
                cleanable = False
                break
        if (cleanable):
            list_cov_poles.remove(sp)
            print("\nremoving ", sp)
            for x in (meters_by_sp):
                cov_meters[int(x)] -= 1
                #print(x, "   ", cov_meters[x])
    #print(time.time()-st)
    #print(cov_meters)
    print(np.sum(cov_meters != 0))
print((time.time() - start_time))
print(len(list_cov_poles))
print(total)
#print(list_cov_poles)
