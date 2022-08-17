from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

def model1(x,a,b,c,d,e,f):
    # return a * pow(x, 2) + b * x + c
    # return a * pow(x, 3) + b * pow(x, 2) + c * x + d
    # return a * pow(x, 4) + b * pow(x, 3) + c * pow(x, 2) + d * x + e
    return a * pow(x, 5) + b * pow(x, 4) + c * pow(x, 3) + d * pow(x, 2) + e * x + f

def model2(x,a,b,c,d):
    return 1/ (a * x + b) + c * x + d
    # return a * pow(x, 4) + b * pow(x, 3) + c * pow(x, 2) + d*x + e * pow(x, 5) + f * pow(x, 6) + g
    # return a * pow(x, 4) + b * pow(x, 3) + c * pow(x, 2) + d*x + e * 1/pow(x, 2) + f #* 1/pow(x, 2) + g

# Test
# Load data
filename = 'assignment1/partA/calibration.csv'
IR3Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,7))

distance, voltage = IR3Data.T
peak = 700
distance1 = distance[0:peak+1]
distance2 = distance[peak:-1]
voltage1 = voltage[0:peak+1]
voltage2 = voltage[peak:-1]

plot2Start = [distance2[0], voltage2[0]]
print(plot2Start)

#create the weighting array
y_weight = np.empty(len(voltage1))
#high pseudo-sd values, meaning less weighting in the fit
y_weight.fill(10)
#low values for point 0 and the last points, meaning more weighting during the fit procedure 
y_weight[0] = y_weight[-1] = 0.1

# popt, pcov = curve_fit(func, x_arr, y_arr, p0=(y_arr[0], 1, 1), sigma = y_weight, absolute_sigma = True)

# params1, cov = curve_fit(model1,distance1, voltage1, p0=(voltage1[0], 1), sigma = y_weight, absolute_sigma = True)
params1, cov = curve_fit(model1,distance1, voltage1)
IRfit1 = model1(distance1, *params1)
params2, cov = curve_fit(model2,distance2, voltage2)
IRfit2 = model2(distance2, *params2)

IRfit = np.concatenate([IRfit1, IRfit2])
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
#         
# params, cov = curve_fit(model,fittedx, fittedy)
# fittedIRfit = model(fittedx,*params)
# fittedIRerror = fittedy - fittedIRfit

# fig, axes = subplots(3)
# fig.suptitle('Refitted IR3 plots ')

# axes[0].plot(fittedx, fittedy, '.')
# axes[0].plot(fittedx, fittedIRfit, '.' )
# axes[0].set_title('IR3 raw data')
#  
# axes[1].plot(fittedx, fittedIRerror, '.')
# axes[1].set_title('measurement error')

# axes[2].hist(fittedIRerror, bins=100)
# axes[2].set_title('histogram')

show()


