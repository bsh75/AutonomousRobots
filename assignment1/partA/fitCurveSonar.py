from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

def model(x,a,b):
    return a * x + b

# Test
# Load data
filename = 'assignment1/partA/calibration.csv'
S1Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,8))
distance, rawS1 = S1Data.T

params, cov = curve_fit(model, distance, rawS1)
fit = model(distance,*params)
error = rawS1 - fit

# Plot original data
fig, axes = subplots(2)
fig.suptitle('Calibration Sonar 1')
axes[0].plot(distance, rawS1, '.', alpha=0.2)
axes[0].plot(distance, fit)
axes[0].set_title('Plot plus fit')
axes[1].plot(distance, error, '.', alpha=0.2)
axes[1].set_title('error')

# 1st iteration
n = 0 # index for outer loop
N = 4 # How many times the data is cleaned
errorMin = -0.3 # Eyeballed min value that appears on error plot
errorMax = 0.7 # Eyeballed max value

while n <= N:
    n += 1
    if n == 3:
        errorMin = -0.1 # Eyeballed min val on second to last error plot
        errorMax = 0.1 # Eyeballed max val on second to last error plot
    # Convert back to list
    distance = distance.tolist()
    error = error.tolist()
    rawS1 = rawS1.tolist()

    # Remove outliers
    i = 0
    while i < len(error):
        # print(error[i])
        if error[i] > errorMax:
            distance.pop(i)
            rawS1.pop(i)
            error.pop(i)
        elif error[i] < errorMin:
            distance.pop(i)
            rawS1.pop(i)
            error.pop(i)
        i += 1
    # Convert back to array for plotting
    distance = np.array(distance)
    rawS1 = np.array(rawS1)   

    params, cov = curve_fit(model, distance, rawS1)
    fit = model(distance,*params)
    error = rawS1 - fit

fig2, axes = subplots(3)
fig2.suptitle('Calibration Sonar 1 (Second)')
axes[0].plot(distance, rawS1, '.', alpha=0.2)
axes[0].plot(distance, fit)
axes[0].set_title('Plot plus fit')
axes[1].plot(distance, error, '.', alpha=0.2)
axes[1].set_title('error')
axes[2].hist(error, bins=100)
axes[2].set_title('histogram')

print(params)
print(np.std(error), np.mean(error))

plt.show()