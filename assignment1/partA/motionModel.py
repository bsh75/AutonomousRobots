# from math import dist
from numpy import loadtxt, gradient
from matplotlib.pyplot import plot, subplots, show
from scipy.optimize import curve_fit
import statistics



 # Model for fit
def model(x,a,b):
    return a * x + b

# Load data
filename = 'assignment1/partA/training1.csv'
#filename = 'assignment1/partA/training2.csv'
data = loadtxt(filename, delimiter=',', skiprows=1, usecols = (1,2,3))
time, distance, v_com = data.T # This should be distance variable not dist funtion?? see top line (think was automatically added)

### Find relationship between commanded and measured
# Get list of dts and velocity estimates (Same as michael did in Speeds Plot)
dt = time[1:] - time[0:-1]
v_est = gradient(distance, time)

# Plot to show linear relationship then find fit
fig, axes = subplots(2)
params, cov = curve_fit(model, v_com, v_est)
fit = model(v_com,*params)
axes[0].plot(time, v_est,)
axes[0].plot(time, v_com)
axes[0].set_xlabel('Commanded Velocity (m/s)')
axes[0].set_ylabel('Measured Velocity (m/s)')
# Gives the equations between the commanded and measured speeds
a = params[0]
b = params[1]
print(a, b)
v_mod = [1/a*i+b for i in v_est]
#print(d)
axes[1].plot(time, v_mod)
axes[1].plot(time, v_com)

error = v_com - v_mod
fig,axes = subplots(2)
axes[0].plot(time,error, '.')
axes[1].hist(error, bins = 100)

set1Mean = statistics.mean(error)
set1Variance = statistics.variance(error, set1Mean)

print("Training set1 mean error = {0:.5f} with variance {1:.5f}". format(set1Mean,set1Variance))
print("Training set1 mean error = {} with variance {}". format(set1Mean,set1Variance))

show()

    

