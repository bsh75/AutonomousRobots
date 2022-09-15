from symfit import parameters, variables, Fit, Piecewise, exp, Eq, Model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from numpy import loadtxt

filename = 'assignment1/partA/calibration.csv'
IR3Data = loadtxt(filename, delimiter=',',skiprows=1, usecols=(2,7))
dist, ir4_raw = IR3Data.T

# Remove data outside sensor range
ir4Min = 1
ir4Max = 5
xdata = []
ydata = []

for i in range(0, len(dist)):
    if dist[i] >= ir4Min and dist[i] <= ir4Max:
        xdata.append(dist[i])
        ydata.append(ir4_raw[i])

xdata = np.array(xdata)
ydata = np.array(ydata)


x, y = variables('x, y')
a, b, c, d, e, f, g, h, i, j, x0 = parameters('a, b, c, d, e, f, g, h, i, j, x0')

# Make a piecewise model
# y1 = a * x**3 + b * x**2 + c * x + d
y1 = a * x**4 + b * x**3 + c * x**2 + d * x + e
# y1 = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
y2 = 1/ (g* x + h) + i * x + j
model = Model({y: Piecewise((y1, x <= x0), (y2, x > x0))})
# As a constraint, we demand equality between the two models at the point x0
# to do this, we substitute x -> x0 and demand equality using `Eq`
# constraints = [
#     Eq(y1.subs({x: x0}), y2.subs({x: x0}))
# ]

constraints = [
    Eq(y1.diff(x).subs({x: x0}), y2.diff(x).subs({x: x0})),
    Eq(y1.subs({x: x0}), y2.subs({x: x0}))
]
# Generate example data
# xdata = np.linspace(-4, 4., 50)
# ydata = model(x=xdata, a=0.0, b=1.0, x0=1.0).y
# np.random.seed(2)
# ydata = np.random.normal(ydata, 0.5)  # add noise
# xdata = distance
# ydata = voltage

# Help the fit by bounding the switchpoint between the models
x0.min = 0.7
x0.max = 0.9

fit = Fit(model, x=xdata, y=ydata, constraints=constraints)
fit_result = fit.execute()
yFit = model(x=xdata, **fit_result.params).y

error = ydata - yFit

mean = np.mean(error)
var = np.var(error)

print("Old Mean =", mean, "Old Var =", var)

fig1, axes = subplots(2)
axes[0].plot(xdata, ydata, '.')
axes[0].plot(xdata, yFit)

ydataFull = ydata
tolerance = 0.5
n = 0
N = 2

while n < N:
    n += 1
    xdata = xdata.tolist()
    ydata = ydata.tolist()

    k = 0
    while k < len(xdata):
        if (abs(error[k]) > tolerance):
            xdata.pop(k)
            ydata.pop(k)
        k += 1

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    fit = Fit(model, x=xdata, y=ydata, constraints=constraints)
    fit_result = fit.execute()
    yFit = model(x=xdata, **fit_result.params).y

error = ydata - yFit

fig2, axes2 = subplots(2)
axes2[0].plot(xdata, ydata, '.')
axes2[0].plot(xdata, yFit)
axes2[1].hist(error, bins=100)

mean = np.mean(error)
var = np.var(error)

print("Mean =", mean, " Var =", var)
# print(fit_result)

########### Get Lookup tables for variances ################
# Split error into N lists depending on the xList values assosiated
def divide_chunks(xL, errorL, N):
    """Function takes a list of errors and their assosiated positions in
    x axis and returns a list of x divisions and the variances assosiated 
    with each division"""
    print("original list size of: {}".format(len(xL)))
    splitList = []
    splitError = []
    # errors = []
    div = (max(xL)-min(xL))/N
    print("divider = {}".format(div))
    for n in range(0, N):
        xSection = []
        errorSection = []
        for i in range(0, len(xL)):
            if xL[i] >= (min(xL) + n*div) and xL[i] < (min(xL) + (n+1)*div):
                xSection.append(xL[i])
                errorSection.append(errorL[i])
        splitList.append(xSection)
        splitError.append(errorSection)

    # sum = 0
    # for i in range(0, len(splitList)):
    #     size = len(splitList[i])
    #     sum += size
    # print("sum of sections: {}".format(sum))
    vars = []
    xdivisions = np.linspace(min(xL), max(xL), N)
    for e in splitError:
        vars.append(np.var(e))
    print("x lookup list is {}".format(xdivisions))
    print("associated variances in error: {}".format(vars))
    return xdivisions, vars

n = 5
X, E = divide_chunks(xdata, error, n)

plt.show()