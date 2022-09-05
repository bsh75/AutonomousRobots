from tkinter import Y
import numpy as np
from numpy import loadtxt



filename = 'assignment1/partA/test.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

# Motion Model (form motionCombined.py)
a = 0.858707
b = -0.001387
def motion(x):
    return a * x + b

def motionI(z):
    return (z + b)/a

motionMeanE = 0.000228
motionVarE = 0.00053

# IR4 Sensor Model
a = 3.908538
b = 34.46409
c = -74.71687
d = 40.92253
e = -2.220859
f = 1.492627
g = 0.1180388
h = 0.1500129
i = 0.3617102
j = -1.330797
def ir4(x):
    if x <= 0.8:
        y = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    else:
        y = 1/ (g* x + h) + i * x + j
    return y

ir4MeanE = -0.0016598
ir4VarE = 0.0395728

# Sonar Model
a = 0.99499792
b = -0.01731528
def sonar(x,a,b):
    return  a * x + b

def sonarInv(z):
    return (z - b)/a

sonarMeanE = -1.814182e-12
sonarVarE = 0.007945636

# IR3 Model
a = 3.37323446
b = -0.00565075
c = 0.17960171
d = -0.00724171
def ir3(x, a, b, c, d):
    return 1/(a*x + b) + c*x + d

def ir3Inv(z):
    return 

ir3MeanE = 6.8549e-10
ir3VarE = 0.005288364
