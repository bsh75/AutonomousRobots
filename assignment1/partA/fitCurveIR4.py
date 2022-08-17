from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

def model(x,a,b,c,d,e,f,g):
    return a * pow(x, 4) + b * pow(x, 3) + c * pow(x, 2) + d*x + e * pow(x, 5) + f * pow(x, 6) + g
    # return a * pow(x, 4) + b * pow(x, 3) + c * pow(x, 2) + d*x + e * 1/pow(x, 2) + f #* 1/pow(x, 2) + g

# Test
# Load data
filename = 'assignment1/partA/calibration.csv'
IR3Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,7))

distance, voltage = IR3Data.T

params, cov = curve_fit(model,distance, voltage)
print(params)
IRfit = model(distance, *params)
IRerror = voltage - IRfit

fig, axes = subplots(2)
fig.suptitle('Calibration IR3 ')

axes[0].plot(distance, voltage, '.')
axes[0].plot(distance, IRfit, '.' )
axes[0].set_title('IR3 raw data')

axes[1].plot(distance, IRerror, '.')
axes[1].set_title('measurement error')

# axes[2].hist(IRerror, bins=100)
# axes[2].set_title('histogram')


# upperLim = 0.1
# lowerLim = -0.1
# fittedx = []
# fittedy = []
# print(len(x))
# print(len(y))
# for i in range(len(IRerror)):
#     if (IRerror[i] < upperLim) & (IRerror[i] > lowerLim):
#         fittedx.append(x[i])
#         fittedy.append(y[i])
#     else:
#         pass

# fittedx = np.array(fittedx)
# fittedy = np.array(fittedy)
        

# params, cov = curve_fit(model,fittedx, fittedy)
# fittedIRfit = model(fittedx,*params)
# fittedIRerror = fittedy - fittedIRfit

# fig, axes = subplots(3)
# fig.suptitle('Refitted IR3 plots ')

# axes[0].plot(fittedx, fittedy, '.')
# axes[0].plot(fittedx, fittedIRfit, '.' )
# axes[0].set_title('IR3 raw data')


# axes[1].plot(fittedx, fittedIRerror, '.')
# axes[1].set_title('measurement error')

# axes[2].hist(fittedIRerror, bins=100)
# axes[2].set_title('histogram')

show()


