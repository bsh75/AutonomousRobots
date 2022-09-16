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

distanceAll = distance
rawS1All = rawS1

n = 0 # index for outer loop
N = 10 # How many times the data is cleaned
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
        i += 1

    distance = np.array(distance)
    rawS1 = np.array(rawS1)
    # Replot and refit
    params, cov = curve_fit(model, distance, rawS1)
    fit = model(distance,*params)
    error = fit - rawS1

# Plot cleaned data with fit
fig, axes = subplots(3)
fig.tight_layout()
axes[0].set_title('Sonar')
axes[0].plot(distanceAll, rawS1All, '.', alpha=0.2)
axes[0].plot(distance, rawS1, '.', alpha=0.2, color='green')
axes[0].plot(distance, fit, color='orange')
axes[0].legend(["Outliers", "Points fitted", "Fit"])
axes[0].set_xlabel('Range (m)')
axes[0].set_ylabel('Measured Distance (m)')
axes[1].plot(distance, error, '.', alpha=0.2)
axes[1].set_xlabel('Range (m)')
axes[1].set_ylabel('Error (m)')
axes[2].hist(error, bins=100)
axes[2].set_xlabel('Error (m)')
axes[2].set_ylabel('Counts')

# Show parameters and distribution of error
print(params)
print(np.mean(error), np.var(error))

########### Get Lookup tables for variances ################
#### Split error into N lists depending on the xList values assosiated ############
def divide_chunks(xL, errorL, N):
    """Function takes a list of errors and their assosiated positions in
    x axis and returns a list of x divisions and the variances assosiated 
    with each division"""
    print("original list size of: {}".format(len(xL)))
    splitList = []
    splitError = []
    # errors = []
    div = (max(xL)-min(xL))/N
    print("divider = {}".format(div))
    for n in range(0, N-1):
        xSection = []
        errorSection = []
        for i in range(0, len(xL)):
            if xL[i] >= (min(xL) + n*div) and xL[i] < (min(xL) + (n+1)*div):
                xSection.append(xL[i])
                errorSection.append(errorL[i])
        splitList.append(xSection)
        splitError.append(errorSection)

    # sum = 0
    # for i in range(0, len(splitList)):
    #     size = len(splitList[i])
    #     sum += size
    # print("sum of sections: {}".format(sum))
    vars = []
    xdivisions = np.linspace(min(xL), max(xL), N)
    for e in splitError:
        vars.append(np.var(e))
    print("x lookup list is {}".format(xdivisions))
    print("associated variances in error: {}".format(vars))
    return xdivisions, vars

n = 5
X, E = divide_chunks(distance, error, n)


plt.show()