import numpy as np
import time
# CONSTANTS
NO_METERS_P = 108
NO_POLES_P = 117
NO_METERS_C = 7172
NO_POLES_C = 6550



data_file = np.loadtxt('cap360.txt', dtype=np.int32)
# Create an adjacency matrix of poles and meters
Adj_pm = np.zeros((NO_POLES_C, NO_METERS_C))
# Create an adjacency list for each pole
poles = [[] for i in range(NO_POLES_C)]
# Create an adjacency list for each meter
meters = [[] for i in range(NO_METERS_C)]

for x in data_file:
    if (x[1]-1 < 0):
        print("sime")
        print(x)
    x[0] -= 1 # Converting to 0-index
    x[1] -= 1
    Adj_pm[x[1]][x[0]] = 1
    meters[x[0]].append(x[1])
    poles[x[1]].append(x[0])


start_time = time.time()
sorted_array = np.argsort([len(x) for x in poles])[::-1]
# Create an empty for covered meters
cov_meters = np.zeros(NO_METERS_C)
# list of covered meters
list_cov_meters = []
# list of covering poles
list_cov_poles = []
j = 0
#while (np.any(cov_meters == 0)):
for j in range(len(sorted_array)):
    best_pole = sorted_array[j] # best pole by length
    list_cov_poles.append(best_pole)
    for x in (poles[best_pole]):
        cov_meters[x] += 1
        #if x not in list_cov_meters:
         #   list_cov_meters.append(x)
    #j += 1
    # Clean up
    for i in range(0, len(list_cov_poles), 1): # for each already selected pole(sp)
        if (i >= len(list_cov_poles)):
            break
        sp = list_cov_poles[i]
        cleanable = True
        for x in (poles[sp]):
            if ((cov_meters[x] - 1) == 0):
                cleanable = False
                break
        if (cleanable):
            list_cov_poles.remove(sp)
            #print("\nremoving ", sp)
            for x in (poles[sp]):
                cov_meters[x] -= 1
                #print(x, "   ", cov_meters[x])
           #print("----------------------")
print((time.time() - start_time))
print(len(list_cov_poles))
print(list_cov_poles)
