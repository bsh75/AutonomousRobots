from math import dist
from numpy import loadtxt, gradient
from matplotlib.pyplot import subplots, show

# Load data
filename = 'assignment1/partA/training1.csv'
#filename = 'assignment1/partA/training2.csv'

data = loadtxt(filename, delimiter=',', skiprows=1, usecols = (1,2,3))

time, dist, v_com = data.T

#defining lists the length of dats imported - 
xbelief = [0]*len(time)
g_ux=[0]*len(time)
wn = [0]*len(time)
time = time.tolist()
v_com = v_com.tolist()




for i  in range(len(time)-1):
    #Motion model somewhat works, error does accumulate
    #Question for micheal or you (brett). are we modelling velocity or displacement. (work up until now has been for displacement )
    t_step = time[i+1] - time[i]
    g_ux[i] = v_com[i]*t_step
    xbelief[i+1] = xbelief[i] + g_ux[i] 
    #code for trying to add process noise following formula in guide. Greened me out as its negative of the motion model 

    #wn[i+1] = xbelief[i+1]-xbelief[i] - g_ux[i] tryingto add sensor noise model in, obviously not right eway to go about it 
    #xbelief[i+1] = xbelief[i] + g_ux[i] - wn[i+1]
    
    
    
    

fig, axes = subplots(1)
axes.plot(time, dist, label='measured distance')
axes.plot(time, xbelief, label='modelled distance')
axes.set_xlabel('Time')
axes.set_ylabel('displacment (m/s)')
axes.legend()

show()

    

