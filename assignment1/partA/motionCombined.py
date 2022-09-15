from math import dist
import numpy as np
from numpy import loadtxt, gradient
from matplotlib.pyplot import plot, subplots, show
from scipy.optimize import curve_fit


 # Model for fit
def model(x,a,b):
    return a * x + b

# Load data from each training file
filename1 = 'assignment1/partA/training1.csv'
data1 = loadtxt(filename1, delimiter=',', skiprows=1, usecols = (1,2,3))
time1, distance1, v_com1 = data1.T

filename2 = 'assignment1/partA/training2.csv'
data2 = loadtxt(filename2, delimiter=',', skiprows=1, usecols = (1,2,3))
time2, distance2, v_com2 = data2.T

### Find relationship between commanded and measured
# get the estimated velocity from the measured displacement data
dt = time1[1:] - time1[0:-1]
v_est1 = gradient(distance1, time1)

dt2 = time2[1:] - time2[0:-1]
v_est2 = gradient(distance2, time2)

# Concatenate the data to find the overall relationship
v_est = np.concatenate([v_est1, v_est2])
v_com = np.concatenate([v_com1, v_com2])

# Find fit
params, cov = curve_fit(model, v_com, v_est)
fit = model(v_com,*params)
errorPerV = fit - v_est
a = params[0]
b = params[1]
print("a =", a, " b =", b)

# Find velocity model by back solving using the relationship just found
v_mod = [1/a*i-b for i in v_est] 
error = v_com - v_mod
# Plot results
fig, axes = subplots(3)
axes[0].plot(v_com, v_est, '.')
axes[0].plot(v_com, fit)
axes[0].set_xlabel('Commanded Velocity (m/s)')
axes[0].set_ylabel('Measured Velocity (m/s)')

axes[1].hist(error, bins=100)
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Error (m/s)')

axes[2].plot(v_com, error, '.')
axes[2].set_xlabel('Commanded Velocity (m/s)')
axes[2].set_ylabel('Error (m/s)')

fig, axes = subplots(2)
axes[0].plot(time1, v_com1)
axes[0].plot(time1, v_mod[0:4819])
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Velocity (m/s)')

axes[1].plot(time2, v_com2)
axes[1].plot(time2, v_mod[4819:])
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (m/s)')

# Get mean and variance of new motion model
mean = np.mean(error)
var = np.var(error)

print("Mean =", mean, " Var =", var)

########### Get Lookup tables for variances ################
# # Split error into N lists depending on the xList values assosiated
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
X, E = divide_chunks(v_com, error, n)


show()
