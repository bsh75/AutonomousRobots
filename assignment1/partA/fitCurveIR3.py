from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit
import statistics

def model(x,a,b,c,d):
    return  1/(a*x+b) + c*x + d # Alfies old: 0.5*a * np.exp(-0.5*b *x) + c*x +d

# Test
# Load data
filename = 'assignment1/partA/calibration.csv'
IR3Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,6))

x, y = IR3Data.T

params, cov = curve_fit(model,x, y)
IRfit = model(x,*params)
IRerror = y - IRfit

IRerrorMean = statistics.mean(IRerror)
IRerrorVariance = statistics.variance(IRerror,IRerrorMean)
print("Original IR3 mean error = {0:.4f} with variance {1:.4f}". format(IRerrorMean,IRerrorVariance))
print("Original IR3 mean error = {} with variance {}". format(IRerrorMean,IRerrorVariance))

fig, axes = subplots(3)
fig.suptitle('Calibration IR3 ')

axes[0].plot(x, y, '.')
axes[0].plot(x, IRfit, '.' )
axes[0].set_title('IR3 raw data')


axes[1].plot(x, IRerror, '.')
axes[1].set_title('measurement error')

axes[2].hist(IRerror, bins=100)
axes[2].set_title('histogram')


maxVar = 0.3
fittedx = []
fittedy = []
print(len(x))
print(len(y))
for i in range(len(IRerror)):
    if abs(IRerror[i]) < maxVar:
        fittedx.append(x[i])
        fittedy.append(y[i])
    else:
        pass

fittedx = np.array(fittedx)
fittedy = np.array(fittedy)
        

params, cov = curve_fit(model,fittedx, fittedy)
fittedIRfit = model(fittedx,*params)
fittedIRerror = fittedy - fittedIRfit

set1Mean = np.mean(fittedIRerror)
set1Variance = np.var(fittedIRerror)
print("Refitted mean error = {0:.4f} with variance {1:.4f}". format(set1Mean,set1Variance))
print("Refitted mean error = {} with variance {}". format(set1Mean,set1Variance))



fig, axes = subplots(3)
fig.suptitle('Refitted IR3 plots ')

axes[0].plot(fittedx, fittedy, '.')
axes[0].plot(fittedx, fittedIRfit, '.' )
axes[0].set_title('IR3 raw data')


axes[1].plot(fittedx, fittedIRerror, '.')
axes[1].set_title('measurement error')

axes[2].hist(fittedIRerror, bins=50)
axes[2].set_title('histogram')



show()


