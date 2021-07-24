from sipy import differential_evolution
from numpy import (array, transpose, dot, exp, argmax, maximum, minimum, ones,
                   hstack)
from numpy import sum as mysum
from numpy.random import rand
from castar import (numbneig, numbstates, numbvars, lb, ub, x0, ndata, ds, a,
                    alfa, namebestsolfile, nameant, itersopt)

loadant = False

iters = itersopt

if loadant:
    thetas2 = []
    with open(nameant, "r") as file:
        for line in file.readlines():
            thetas2.append(float(line.strip()))
    thetas2 = array(thetas2)
    print(f"Loaded data from {nameant}")


def sigmoid(z):
    return 1/(1+exp(-z))


def nn2(X, Theta1, Theta2):
    Z2 = dot(hstack((X, ones([ndata, 1]))), transpose(Theta1))
    Z2 = sigmoid(Z2)
    y = dot(hstack((Z2, ones([ndata, 1]))), transpose(Theta2))
    sol = argmax(y, axis=1)
    return sol


def myerror(thetas):
    theta1 = thetas[:a*(numbneig)].reshape(numbstates,numbneig)
    theta2 = thetas[(numbneig)*a:(numbneig)*a+numbstates*(a+1)].reshape(numbstates, a+1)
    return mysum(ds != nn2(x0, theta1, theta2))


bounds = [(lb, ub)] * numbvars

print("Starting Training ...")

if loadant:

    mylb = [lb] * numbvars
    myub = [ub] * numbvars

    initpop = minimum(maximum(thetas2 * (1 + alfa * 2 * (rand(len(thetas2)) - 0.5)), mylb), myub)

    result = differential_evolution(myerror, bounds, maxiter=iters, popsize=15, disp=True, init='random', x0=initpop)

else:

    result = differential_evolution(myerror, bounds, maxiter=iters, popsize=15, disp=True, init='random')

xopt = result.x
fopt = result.fun

print("Training Complete!")

thetas2 = xopt

print(myerror(thetas2))
print(fopt)

with open(namebestsolfile, "w") as file:
    for theta in xopt:
        file.write(f"{str(theta)}\n")

print(f"Best Solution saved at {namebestsolfile}")

theta1 = thetas2[:a*(numbneig)].reshape(numbstates,numbneig)
theta2 = thetas2[(numbneig)*a:(numbneig)*a+numbstates*(a+1)].reshape(numbstates, a+1)

print(x0)
print(theta1)
print(theta2)
print(nn2(x0, theta1, theta2))
