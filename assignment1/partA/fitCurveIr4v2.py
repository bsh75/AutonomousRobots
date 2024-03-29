from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit
import statistics

def model(x,a,b,c,d):
    return  1/(a*x+b) + c*x + d # Alfies old: 0.5*a * np.exp(-0.5*b *x) + c*x +d

filename = 'assignment1/partA/calibration.csv'
IR4Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,7))
dist, ir4_raw = IR4Data.T

# Remove data outside sensor range
ir4Min = 1
ir4Max = 5
x = []
y = []

for i in range(0, len(dist)):
    if dist[i] >= ir4Min and dist[i] <= ir4Max:
        x.append(dist[i])
        y.append(ir4_raw[i])

x = np.array(x)
y = np.array(y)

params, cov = curve_fit(model,x, y)
IRfit = model(x,*params)
IRerror = y - IRfit

IRerrorMean = statistics.mean(IRerror)
IRerrorVariance = statistics.variance(IRerror,IRerrorMean)
print("Original IR4 mean error = {0:.4f} with variance {1:.4f}". format(IRerrorMean,IRerrorVariance))
print("Original IR4 mean error = {} with variance {}". format(IRerrorMean,IRerrorVariance))

fig, axes = subplots(3)
fig.suptitle('Calibration IR4 ')

axes[0].plot(x, y, '.')
axes[0].plot(x, IRfit, '.' )
axes[0].set_title('IR4 raw data')

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
fig.suptitle('Refitted IR4 plots ')

axes[0].plot(fittedx, fittedy, '.')
axes[0].plot(fittedx, fittedIRfit, '.' )
axes[0].set_title('IR4 raw data')


axes[1].plot(fittedx, fittedIRerror, '.')
axes[1].set_title('measurement error')

axes[2].hist(fittedIRerror, bins=50)
axes[2].set_title('histogram')

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
X, E = divide_chunks(fittedx, fittedIRerror, n)
print(len(X), len(E))

show()


