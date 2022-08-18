from numpy import loadtxt, gradient
from matplotlib.pyplot import subplots, show

# Load data
filename = 'assignment1/partA/training1.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

v_com = velocity_command

dt = time[1:] - time[0:-1]
v_est = gradient(distance, time)

fig, axes = subplots(1)
axes.plot(v_com,v_est,'.', label='measured speed v commanded speed')
axes.set_ylabel('measured speed')
axes.set_xlabel('commanded speed')
axes.legend()

show()
