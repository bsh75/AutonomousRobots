from tkinter import W, Y
import numpy as np
from numpy import loadtxt
from matplotlib.pyplot import plot, subplots, show

filename = 'assignment1/partA/training1.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, dist, v_comm, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

##### Lookup Function and Tables ##########################
# Lookup tables attained from 'divide_chunks' function in models
xLookupIr3 = [0.09679938, 0.90525033, 1.71370128, 2.52215223, 3.33060318]
variancesIr3 = [0.004826065829283602, 0.004357475760755817, 0.005871319309306362, 0.004765047580434493, 0.006197931690626443]
xLookupIr4 = [0.09679938, 0.90525033, 1.71370128, 2.52215223, 3.33060318]
variancesIr4 = [0.12220775598917524, 0.015966623766261956, 0.012969067620212034, 0.019064140416456637, 0.022062251790182192]
xLookupSonar = [0.09679938, 0.90525033, 1.71370128, 2.52215223, 3.33060318]
variancesSonar = [2.281659850515315e-05, 1.7174942189593155e-05, 3.873231279068888e-05, 0.00011085378539295636, 0.00013254441517565623]
xLookupMotion = [-0.53948363, -0.27169283, -0.00390204,  0.26388876,  0.53167956]
variancesMotion = [0.0019200491161449286, 0.00123474268792669, 0.0002559447356770384, 0.0009352631546926811, 0.0014164489783990344]

def VarLookup(x, xLookup, variances):
    #if ((x >= min(xLookup)) and (x < max(xLookup))):
        
    for i in range(0, len(xLookup)-1):
        # print(x, xLookup[i], xLookup[i+1])
        if (x >= xLookup[i]) and (x < xLookup[i+1]):
            # print("Var = {}".format(variances[i]))

            Var = variances[i]
    return Var

##### Motion Model (form motionCombined.py) ##################
aM = 0.858707
bM = -0.001387

def motion(v_com, prevPos, dt):
    v = aM * v_com + bM
    newPos = prevPos + v * dt
    print(v_com)
    # print(" new pos = {}".format(newPos))
    # print(" prev pos = {}".format(prevPos))
    # print(" v_com = {}".format(v_com))
    # print(dt)

    return newPos

motionMeanE = 0.000228
motionVarE = 0.00053

def linearMotionVar(x0, VarPrevious):
    processVar = motionVarE #VarLookup(x0, xLookupMotion, variancesMotion)
    motionVarX = aM**2 * VarPrevious + processVar # Var(aX + b) = a^2*Var(X)
    return motionVarX

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
a4 = 46.08906 
b4 = -88.06964 
c4 = 47.97408 
d4 = -3.771997
e4 = 1.578568
g4 = 2.129193e-01 
h4 = 1.632003e-01 
i4 = 1.971881e-01 
j4 = -1.051197e-01 
x04 = 0.8

ir4MeanE = -0.0016598
ir4VarE = 0.0395728

def ir4(x):
    if x <= x04:
        z = a4 * x**4 + b4 * x**3 + c4 * x**2 + d4 * x + e4
    else:
        z = 1/ (g4* x + h4) + i4 * x + j4
    return z

def ir4Inv(z, xPast):
    """returns a list of possible x values for a given z"""
    possibleX = []
    # Find the roots of the first section of function and add them if the suitable
    roots = np.roots([a4, b4, c4, d4, e4-z])    
    for root in roots:
        if ((root > 0 and root < x04) and (root.imag == 0)):
            possibleX.append(root.real)
    # Find roots of second section and add them if they suitable (equations from wolfram alpha)
    endx1 = (-g4*j4 + g4*z - h4*i4)/(2*g4*i4) + np.sqrt((-g4*j4 + g4*z - h4*i4)**2 + 4*g4*i*(-h4*j4 + h4*z -1))/(2*g4*i4)
    endx2 = (-g4*j4 + g4*z - h4*i4)/(2*g4*i4) - np.sqrt((-g4*j4 + g4*z - h4*i4)**2 + 4*g4*i4*(-h4*j4 + h4*z -1))/(2*g4*i4)
    if (endx1 > x04 and endx1 < 3.5):
        possibleX.append(endx1)
    if (endx2 > x04 and endx2 < 3.5):
        possibleX.append(endx2)
    # Make sure that the estimate returned is the one closest to past estimate
    X = xPast
    for x in possibleX:
        if abs(xPast - x) < abs(xPast - X):
            X = x
    return X

def derivativeIr4(x):
    if x <= x04:
        return 4*a4*x**3 + 3*b4*x**2 + 2*c4*x + d4
    else:
        return -a4/(a4*x + b4)**2 + c4

def linearIr4Var(x0):
    """Function linearises sensor model about x0 and calculates variance in Z based on linear model"""
    # Taylor series with 2 terms [ h = h(x0)) + h'(x0)(x-x0) ] {ir3Inv(x0) + derivativeIr3(x0)*(x-x0)}
    # turned into y = mx+k form:
    varZ = ir4VarE #VarLookup(x0, xLookupIr4, variancesIr4)
    m = derivativeIr4(x0)
    varX = varZ/m**2
    return varX


##### Sonar Model ########################################
aS = 0.99499792
bS = -0.01731528
sonarMeanE = -1.814182e-12
sonarVarE = 0.007945636

def sonar(x):
    return  aS * x + bS

def sonarInv(z):
    return (z - bS)/aS

def linearSonarVar(x0):
    varZ = sonarVarE #VarLookup(x0, xLookupSonar, variancesSonar)
    sonarVarX = varZ/aS**2  # Var(aX + b) = a^2*Var(X)
    return sonarVarX

##### IR3 Model ###########################################
a3 = 3.37323446
b3 = -0.00565075
c3 = 0.17960171
d3 = -0.00724171

ir3MeanE = 6.8549e-10
ir3VarE = 0.005288364

def ir3(x):
    return 1/(a3*x + b3) + c3*x + d3

def ir3Inv(z, xPast):
    """Take a measurement and past estimate and return estimate closest to last"""
    x1 = (-a3*d3 + a3*z - b3*c3)/(2*a3*c3) + np.sqrt((-a3*d3 + a3*z - b3*c3)**2 + 4*a3*c3*(-b3*d3 + b3*z -1))/(2*a3*c3)
    x2 = (-a3*d3 + a3*z - b3*c3)/(2*a3*c3) - np.sqrt((-a3*d3 + a3*z - b3*c3)**2 + 4*a3*c3*(-b3*d3 + b3*z -1))/(2*a3*c3) 
    if abs(xPast - x1) < abs(xPast - x2):
        return x1
    else:
        return x2

def derivativeIr3(x):
    return -a3/(a3*x + b3)**2 + c3

def linearIr3Var(x0):
    """Function linearises sensor model about x0 and calculates variance in Z based on linear model"""
    # Taylor series with 2 terms [ h = h(x0)) + h'(x0)(x-x0) ] {ir3Inv(x0) + derivativeIr3(x0)*(x-x0)}
    # turned into y = mx+k form:
    varZ = ir3VarE #VarLookup(x0, xLookupIr3, variancesIr3)
    m = derivativeIr3(x0)
    varX = varZ/m**2
    return varX
    

##### Testing #############################################
# zS = 0.1
# z4 = 3
# z3 = 0.5
# print("SONAR reading of {} gives estimation of {}".format(zS, sonarInv(zS)))
# print("IR4 reading of {} gives estimation of {}".format(z4, ir4Inv(z4)))
# print("IR3 reading of {} gives estimation of {}".format(z3, ir3Inv(z3)))

# Initial Belief
initialX = 0
initialVar = 0
Xir3Past = initialX
Xir4Past = initialX
postX = []
wIr3 = []
wIr4 = []
wSonar = []
wMotion = []
dt = time[1:] - time[0:-1]

priorVar = []

for i in index-1:
    i = int(i)
    ### Predict
    priorX = motion(v_comm[i], initialX, dt[i])
    priorVar.append(linearMotionVar(priorX, initialVar))

    ### Update
    Xir3 = ir3Inv(raw_ir3[i], Xir3Past)
    Xir4 = ir4Inv(raw_ir4[i], Xir4Past)
    Xsonar = sonarInv(sonar1[i])
    # Past estimates used to find singular current estimate
    Xir3Past = Xir3
    Xir4Past = Xir4
    
    VarIr3 = linearIr3Var(Xir3)
    VarIr4 = linearIr4Var(Xir4)
    VarSonar = linearSonarVar(Xsonar)

    # BLUE for combining sensors
    wIr3.append(1/VarIr3/(1/VarIr3 + 1/VarIr4 + 1/VarSonar))
    wIr4.append(1/VarIr4/(1/VarIr3 + 1/VarIr4 + 1/VarSonar))
    wSonar.append(1/VarSonar/(1/VarIr3 + 1/VarIr4 + 1/VarSonar))
    XBlu = (1/VarIr3*Xir3 + 1/VarIr4*Xir4 + 1/VarSonar*Xsonar)/(1/VarIr3 + 1/VarIr4 + 1/VarSonar)
    VarBlu = 1/(1/VarIr3 + 1/VarIr4 + 1/VarSonar)
    
    # BLUE for combining sensor and motion
    wMotion.append(1/priorVar/(1/VarBlu + 1/priorVar))
    postX.append((1/VarBlu*XBlu + 1/priorVar*priorX)/(1/VarBlu + 1/priorVar))
    postVar = 1/(1/VarBlu + 1/priorVar)
    initialX = postX[i]
    initialVar = postVar
    postX.append(priorX)
    initialX = postX[i]
    initialVar = priorVar[i]
    

print(len(time), len(postX))
plot(time, postX)
# plot(time, priorVar)
# fig, axes = subplots(4)
# axes[0].plot(time, wIr3)
# axes[0].set_ylabel('wIr3')
# axes[1].plot(time, wIr4)
# axes[1].set_ylabel('wIr4')
# axes[2].plot(time, wSonar)
# axes[2].set_ylabel('wSonar')
# axes[3].plot(time, wMotion)
# axes[3].set_ylabel('wMotion')


show()