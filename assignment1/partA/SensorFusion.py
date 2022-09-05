from tkinter import Y
import numpy as np
from numpy import loadtxt


filename = 'assignment1/partA/test.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

##### Motion Model (form motionCombined.py) ##################
a = 0.858707
b = -0.001387
def motion(x):
    return a * x + b

motionMeanE = 0.000228
motionVarE = 0.00053

##### IR4 Sensor Model #######################################
# For 5th order
# a = 4.608906e+01
# b = -8.806964e+01
# c = -74.71687
# d = 40.92253
# e = -2.220859
# f = 1.492627
# g = 0.1180388
# h = 0.1500129
# i = 0.3617102
# j = -1.330797
# For 4th order
a = 46.08906 
b = -88.06964 
c = 47.97408 
d = -3.771997
e = 1.578568
g = 2.129193e-01 
h = 1.632003e-01 
i = 1.971881e-01 
j = -1.051197e-01 
x0 = 0.8

def ir4(x):
    if x <= x0:
        z = a * x**4 + b * x**3 + c * x**2 + d * x + e
    else:
        z = 1/ (g* x + h) + i * x + j
    return z

def ir4Inv(z):
    """returns a list of possible x values for a given z"""
    possibleX = []
    # Find the roots of the first section of function and add them if the suitable
    roots = np.roots([a, b, c, d, e-z])    
    for root in roots:
        if ((root > 0 and root < x0) and (root.imag == 0)):
            possibleX.append(root.real)
    # Find roots of second section and add them if they suitable (equations from wolfram alpha)
    endx1 = (-g*j + g*z - h*i)/(2*g*i) + np.sqrt((-g*j + g*z - h*i)**2 + 4*g*i*(-h*j + h*z -1))/(2*g*i)
    endx2 = (-g*j + g*z - h*i)/(2*g*i) - np.sqrt((-g*j + g*z - h*i)**2 + 4*g*i*(-h*j + h*z -1))/(2*g*i)
    if (endx1 > x0 and endx1 < 3.5):
        possibleX.append(endx1)
    if (endx2 > x0 and endx2 < 3.5):
        possibleX.append(endx2)
    return possibleX

ir4MeanE = -0.0016598
ir4VarE = 0.0395728

##### Sonar Model ########################################
a = 0.99499792
b = -0.01731528
def sonar(x):
    return  a * x + b

def sonarInv(z):
    return (z - b)/a

sonarMeanE = -1.814182e-12
sonarVarE = 0.007945636

##### IR3 Model ###########################################
a = 3.37323446
b = -0.00565075
c = 0.17960171
d = -0.00724171
def ir3(x, a, b, c, d):
    return 1/(a*x + b) + c*x + d

def ir3Inv(z):
    x1 = (-a*d + a*z - b*c)/(2*a*c) + np.sqrt((-a*d + a*z - b*c)**2 + 4*a*c*(-b*d + b*z -1))/(2*a*c)
    x2 = (-a*d + a*z - b*c)/(2*a*c) - np.sqrt((-a*d + a*z - b*c)**2 + 4*a*c*(-b*d + b*z -1))/(2*a*c) 
    return x1, x2

ir3MeanE = 6.8549e-10
ir3VarE = 0.005288364

##### Testing #############################################
zS = 0.1
z4 = 3
z3 = 0.5
print("SONAR reading of {} gives estimation of {}".format(zS, sonarInv(zS)))
print("IR4 reading of {} gives estimation of {}".format(z4, ir4Inv(z4)))
print("IR3 reading of {} gives estimation of {}".format(z3, ir3Inv(z3)))