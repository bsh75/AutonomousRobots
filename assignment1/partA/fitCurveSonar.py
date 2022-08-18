from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit
 
 # Model for fit
def model(x,a,b):
    return a * x + b

# Test
# Load data
filename = 'assignment1/partA/calibration.csv'
S1Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,8))
distance, rawS1 = S1Data.T


n = 0 # index for outer loop
N = 5 # How many times the data is cleaned
tolerance = 0.1 # Acceptable range in data

while n <= N:
    n += 1
    distance = distance.tolist()
    rawS1 = rawS1.tolist()

    i = 0
    while i < len(distance):
        if (abs(distance[i]-rawS1[i]) > tolerance):
            distance.pop(i)
            rawS1.pop(i)
        # # print(error[i])
        # if error[i] > errorMax:
        #     distance.pop(i)
        #     rawS1.pop(i)
        #     error.pop(i)
        # elif error[i] < errorMin:
        #     distance.pop(i)
        #     rawS1.pop(i)
        #     error.pop(i)
        i += 1

    distance = np.array(distance)
    rawS1 = np.array(rawS1)

    params, cov = curve_fit(model, distance, rawS1)
    fit = model(distance,*params)
    error = fit - rawS1

# Plot cleaned data with fit
fig, axes = subplots(3)
fig.suptitle('Calibration Sonar 1')
axes[0].plot(distance, rawS1, '.', alpha=0.2)
axes[0].plot(distance, fit)
axes[0].set_title('Plot plus fit')
axes[1].plot(distance, error, '.', alpha=0.2)
axes[1].set_title('error')
axes[2].hist(error, bins=100)
axes[2].set_title('histogram')

# Show parameters and distribution of error
print(params)
print(np.mean(error), np.std(error))

plt.show()