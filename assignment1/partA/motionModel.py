# from math import dist
from numpy import loadtxt, gradient
from matplotlib.pyplot import plot, subplots, show
from scipy.optimize import curve_fit


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
fig, axes = subplots(1)
params, cov = curve_fit(model, v_com, v_est)
fit = model(v_com,*params)
plot(v_com, v_est, '.')
plot(v_com, fit)
axes.set_xlabel('Commanded Velocity (m/s)')
axes.set_ylabel('Measured Velocity (m/s)')
# Gives the equations between the commanded and measured speeds
a = params[0]
b = params[1]
v_mod = [a*i+b for i in v_est]
# result = [item * 10 for item in my_list]
#print(d)
# plot(time, v_com)
# plot(time, v_mod)
# error = fit - rawS1



# #defining lists the length of dats imported - 
# xbelief = [0]*len(time)
# g_ux=[0]*len(time)
# wn = [0]*len(time)
# time = time.tolist()
# v_com = v_com.tolist()

# for i  in range(len(time)-1):
#     #Motion model somewhat works, error does accumulate
#     #Question for micheal or you (brett). are we modelling velocity or displacement. (work up until now has been for displacement )
#     t_step = time[i+1] - time[i]
#     g_ux[i] = v_com[i]*t_step
#     xbelief[i+1] = xbelief[i] + g_ux[i] 
#     #code for trying to add process noise following formula in guide. Greened me out as its negative of the motion model 

#     #wn[i+1] = xbelief[i+1]-xbelief[i] - g_ux[i] tryingto add sensor noise model in, obviously not right eway to go about it 
#     #xbelief[i+1] = xbelief[i] + g_ux[i] - wn[i+1]
    

# fig, axes = subplots(1)
# axes.plot(time, dist, label='measured distance')
# axes.plot(time, xbelief, label='modelled distance')
# axes.set_xlabel('Time')
# axes.set_ylabel('displacment (m/s)')
# axes.legend()

show()

    

