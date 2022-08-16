from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model(x,a,b,c):
    return a * np.exp(-b *x) + c 
# Test
# Load data
filename = 'assignment1/partA/calibration.csv'
IR3Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,6))

x, y = IR3Data.T

params, cov = curve_fit(model,x, y)
yfit = model(x,*params)
yerror = y- yfit


plt.plot(x,y)
plt.plot(x,yfit)
plt.plot(x,yerror)
plt.show()