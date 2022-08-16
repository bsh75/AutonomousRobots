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

fig, axes = subplots(2)
fig.suptitle('Calibration Sonar 1')

axes[0].plot(distance, rawS1, '.', alpha=0.2)
axes[0].plot(distance, fit)
axes[0].set_title('Plot plus fit')

axes[1].plot(distance, error, '.', alpha=0.2)
axes[1].set_title('error')


distance = distance.tolist()
error = error.tolist()
rawS1 = rawS1.tolist()

# print(type(distance))
# print(type(error))
# print(type(rawS1))
# print(distance)
# print(error)
# print(rawS1)

# Remove outliers
print(len(distance))
print(len(error))
print(len(rawS1))

errorMin = -0.3 # Min value that appears on error plot
errorMax = 0.7
i = 0
while i < len(error):
    print(error[i])
    if error[i] > errorMax:
        distance.pop(i)
        rawS1.pop(i)
        error.pop(i)
    elif error[i] < errorMin:
        distance.pop(i)
        rawS1.pop(i)
        error.pop(i)
    i += 1

distance1 = np.array(distance)
rawS11 = np.array(rawS1)   

params, cov = curve_fit(model, distance1, rawS11)
fit = model(distance1,*params)
error1 = rawS11 - fit

print(len(distance1))
print(len(error1))
print(len(rawS11))

fig2, axes = subplots(2)
fig2.suptitle('Calibration Sonar 1')

axes[0].plot(distance1, rawS11, '.', alpha=0.2)
axes[0].plot(distance1, fit)
axes[0].set_title('Plot plus fit1')

axes[1].plot(distance1, error1, '.', alpha=0.2)
axes[1].set_title('error1')

# plt.plot(x,y, '.')
# plt.plot(x,yfit)
# plt.plot(x,yerror, '.')
plt.show()