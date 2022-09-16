from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit
import statistics

def model(x,a,b,c,d):
    """Hyperbolic fit model to be applied"""
    return  1/(a*x+b) + c*x + d

# Test
# Load data
filename = 'assignment1/partA/calibration.csv'
IR3Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,6))

dist, ir3_raw = IR3Data.T

#### Remove data outside sensor range if wanted (Done to form variance lookups)
ir3Max = 4 # set to 0.8 to limit to sensor range
ir3Min = 0 # set to 0.1 to limit to sensor range
x = []
y = []

for i in range(0, len(dist)):
    if dist[i] >= ir3Min and dist[i] <= ir3Max:
        x.append(dist[i])
        y.append(ir3_raw[i])

x = np.array(x)
y = np.array(y)

# First fit
params, cov = curve_fit(model,x, y)
IRfit = model(x,*params)
# Error of first fit
IRerror = y - IRfit
IRerrorMean = statistics.mean(IRerror)
IRerrorVariance = statistics.variance(IRerror,IRerrorMean)
print("Original IR3 mean error = {0:.4f} with variance {1:.4f}". format(IRerrorMean,IRerrorVariance))
print("Original IR3 mean error = {} with variance {}". format(IRerrorMean,IRerrorVariance))

# Plot Original data, fit, and error before outlier removal
fig, axes = subplots(3)
fig.tight_layout()
axes[0].plot(x, y, '.')
axes[0].plot(x, IRfit, '.' )
axes[0].set_title('IR3 raw data')

axes[1].plot(x, IRerror, '.')
axes[1].set_title('measurement error')

axes[2].hist(IRerror, bins=40)
axes[2].set_title('histogram')

# Outlier rejection
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
        
# Refit a new curve 
params, cov = curve_fit(model,fittedx, fittedy)
fittedIRfit = model(fittedx,*params)
# Error of new fit
fittedIRerror = fittedy - fittedIRfit
set1Mean = np.mean(fittedIRerror)
set1Variance = np.var(fittedIRerror)
print("Refitted mean error = {0:.4f} with variance {1:.4f}". format(set1Mean,set1Variance))
print("Refitted mean error = {} with variance {}". format(set1Mean,set1Variance))

# Plot new fit
fig, axes = subplots(3)
fig.tight_layout()
axes[0].set_title('IR3')
axes[0].plot(fittedx, fittedy, '.', alpha=0.2)
axes[0].plot(fittedx, fittedIRfit)
axes[0].legend(["Raw Data", "Fit"])
axes[0].set_xlabel('Range (m)')
axes[1].plot(fittedx, fittedIRerror, '.', alpha=0.2)
axes[1].set_xlabel('Range (m)')
axes[2].hist(fittedIRerror, bins=40)
axes[2].set_xlabel('Error (m)')

########### Get Lookup tables for variances (Commented out cause takes longer to run) ################
#### Split error into N lists depending on the xList values assosiated ############
# def divide_chunks(xL, errorL, N):
#     """Function takes a list of errors and their assosiated positions in
#     x axis and returns a list of x divisions and the variances assosiated 
#     with each division"""
#     print("original list size of: {}".format(len(xL)))
#     splitList = []
#     splitError = []
#     # errors = []
#     div = (max(xL)-min(xL))/N
#     print("divider = {}".format(div))
#     for n in range(0, N-1):
#         xSection = []
#         errorSection = []
#         for i in range(0, len(xL)):
#             if xL[i] >= (min(xL) + n*div) and xL[i] < (min(xL) + (n+1)*div):
#                 xSection.append(xL[i])
#                 errorSection.append(errorL[i])
#         splitList.append(xSection)
#         splitError.append(errorSection)

#     # sum = 0
#     # for i in range(0, len(splitList)):
#     #     size = len(splitList[i])
#     #     sum += size
#     # print("sum of sections: {}".format(sum))
#     vars = []
#     xdivisions = np.linspace(min(xL), max(xL), N)
#     for e in splitError:
#         vars.append(np.var(e))
#     print("x lookup list is {}".format(xdivisions))
#     print("associated variances in error: {}".format(vars))
#     return xdivisions, vars

# n = 5
# X, E = divide_chunks(fittedx, fittedIRerror, n)

show()


