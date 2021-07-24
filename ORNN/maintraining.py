from sipy import differential_evolution
from numpy import (array, transpose, dot, exp, argmax, maximum, minimum, ones,
                   hstack)
from numpy import sum as mysum
from numpy.random import rand
from castar import (numbneig, numbstates, numbvars, lb, ub, x0, ndata, ds,
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


def nn1(X, Theta1):
    y = dot(hstack((X, ones([ndata, 1]))), transpose(Theta1))
    sol = argmax(y, axis=1)
    return sol


def myerror(thetas):
    theta1 = thetas[:(numbstates)*(numbneig)].reshape(numbstates,numbneig)
    return mysum(ds != nn1(x0, theta1))


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

print(x0)
print(thetas2[:(numbstates)*(numbneig)].reshape(numbstates,numbneig))
print(dot(hstack((x0, ones([ndata, 1]))), transpose(thetas2[:(numbstates)*(numbneig)].reshape(numbstates,numbneig))))
print(nn1(x0, thetas2[:(numbstates)*(numbneig)].reshape(numbstates,numbneig)))
