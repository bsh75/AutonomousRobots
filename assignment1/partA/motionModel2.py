# from math import dist
import numpy as np
from numpy import loadtxt, gradient
from matplotlib.pyplot import plot, subplots, show
from scipy.optimize import curve_fit


 # Model for fit
def model(x,a,b):
    return a * x + b

# Load data
filename = 'assignment1/partA/training2.csv'
#filename = 'assignment1/partA/training2.csv'
data = loadtxt(filename, delimiter=',', skiprows=1, usecols = (1,2,3))
time, distance, v_com = data.T # This should be distance variable not dist funtion?? see top line (think was automatically added)

### Find relationship between commanded and measured
# Get list of dts and velocity estimates (Same as michael did in Speeds Plot)
dt = time[1:] - time[0:-1]
v_est = gradient(distance, time)

# Plot to show linear relationship then find fit
params, cov = curve_fit(model, v_com, v_est)
fit = model(v_com,*params)

# Gives the equations between the commanded and measured speeds
a = params[0]
b = params[1]

v_mod = [1/a*i+b for i in v_est]
error = v_com - v_mod
# result = [item * 10 for item in my_list]
#print(d)
# Plot results
fig, axes = subplots(2)
axes[0].plot(v_com, v_est, '.')
axes[0].plot(v_com, fit)
axes[1].set_xlabel('Commanded Velocity (m/s)')
axes[1].set_ylabel('Measured Velocity (m/s)')

axes[1].plot(time, v_com)
axes[1].plot(time, v_mod)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (m/s)')

fig2, axes2 = subplots(2)
axes2[0].plot(time, error) 
axes2[0].set_xlabel('Time (s)')
axes2[0].set_ylabel('Error (m/s)')

axes2[1].hist(error, bins=100)
axes2[1].set_xlabel('Error')
axes2[1].set_ylabel('Count')

mean = np.mean(error)
stdDev = np.std(error)

show()

    

