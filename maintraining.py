from sipy import differential_evolution
from numpy import (array, transpose, dot, exp, argmax, maximum, minimum, ones,
                   hstack, rot90, fliplr)
from numpy import sum as mysum
from numpy.random import rand
from castar import (lx, ly, lb, ub, ca0, numbneig, a, numbstates, numbvars,
                    alfa, namebestsolfile, nameant, itersopt, known)


#print(ca0)

t1 = 1

ds1 = ca0.copy()
ds1[1, 5] = 2

#print(ds1)

t2 = 2

ds2 = ds1.copy()

#print(ds2)

#input("Continue: ")

print(f"numbvars: {numbvars}")
print(f"numbiters: {itersopt}")

loadant = True

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


def rules(x):

    if mysum(x) == 0:
        return 0
    
    elif not mysum(x!=array([0,0,0,0,2,0,2,2,2])):
        return 2
    elif not mysum(x!=array(rot90(array([0,0,0,0,2,0,2,2,2]).reshape(3,3)).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(array([0,0,0,0,2,0,2,2,2]).reshape(3,3))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(rot90(array([0,0,0,0,2,0,2,2,2]).reshape(3,3)))).flat)):
        return 2
    
    elif not mysum(x!=array([0,0,2,0,2,2,0,2,1])):
        return 2
    elif not mysum(x!=array(rot90(array([0,0,2,0,2,2,0,2,1]).reshape(3,3)).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(array([0,0,2,0,2,2,0,2,1]).reshape(3,3))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(rot90(array([0,0,2,0,2,2,0,2,1]).reshape(3,3)))).flat)):
        return 2
    elif not mysum(x!=array(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3)).flat)):
        return 2
    elif not mysum(x!=array(rot90(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3)))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(rot90(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3))))).flat)):
        return 2
    
    elif not mysum(x != array([0, 2, 0, 2, 2, 2, 2, 1, 2])):
        return 2
    elif not mysum(x != array(rot90(array([0, 2, 0, 2, 2, 2, 2, 1, 2]).reshape(3,3)).flat)):
        return 2
    elif not mysum(x != array(rot90(rot90(array([0, 2, 0, 2, 2, 2, 2, 1, 2]).reshape(3,3))).flat)):
        return 2
    elif not mysum(x != array(rot90(rot90(rot90(array([0, 2, 0, 2, 2, 2, 2, 1, 2]).reshape(3,3)))).flat)):
        return 2
    
    elif not mysum(x!=array([0,2,2,0,2,1,2,2,1])):
        return 2
    elif not mysum(x!=array(rot90(array([0,2,2,0,2,1,2,2,1]).reshape(3,3)).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(array([0,2,2,0,2,1,2,2,1]).reshape(3,3))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(rot90(array([0,2,2,0,2,1,2,2,1]).reshape(3,3)))).flat)):
        return 2
    elif not mysum(x!=array(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3)).flat)):
        return 2
    elif not mysum(x!=array(rot90(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3)))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(rot90(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3))))).flat)):
        return 2

    elif not mysum(x != array([2, 2, 2, 2, 1, 2, 2, 1, 2])):
        return 1
    elif not mysum(x != array(rot90(array([2, 2, 2, 2, 1, 2, 2, 1, 2]).reshape(3,3)).flat)):
        return 1
    elif not mysum(x != array(rot90(rot90(array([2, 2, 2, 2, 1, 2, 2, 1, 2]).reshape(3,3))).flat)):
        return 1
    elif not mysum(x != array(rot90(rot90(rot90(array([2, 2, 2, 2, 1, 2, 2, 1, 2]).reshape(3,3)))).flat)):
        return 1
    
    elif not mysum(x!=array([0,2,1,2,2,1,1,1,1])):
        return 2
    elif not mysum(x!=array(rot90(array([0,2,1,2,2,1,1,1,1]).reshape(3,3)).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(array([0,2,1,2,2,1,1,1,1]).reshape(3,3))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(rot90(array([0,2,1,2,2,1,1,1,1]).reshape(3,3)))).flat)):
        return 2
    elif not mysum(x!=array(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3)).flat)):
        return 2
    elif not mysum(x!=array(rot90(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3)))).flat)):
        return 2
    elif not mysum(x!=array(rot90(rot90(rot90(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3))))).flat)):
        return 2
    
    elif not mysum(x != array([2, 1, 2, 2, 1, 2, 1, 1, 1])):
        return 1
    elif not mysum(x != array(rot90(array([2, 1, 2, 2, 1, 2, 1, 1, 1]).reshape(3,3)).flat)):
        return 1
    elif not mysum(x != array(rot90(rot90(array([2, 1, 2, 2, 1, 2, 1, 1, 1]).reshape(3,3))).flat)):
        return 1
    elif not mysum(x != array(rot90(rot90(rot90(array([2, 1, 2, 2, 1, 2, 1, 1, 1]).reshape(3,3)))).flat)):
        return 1
    
    elif not mysum(x != array([2, 1, 2, 1, 1, 1, 2, 1, 2])):
        return 1
    
    return -1


def applyrules(X, y):
    myy = y.copy()
    for indice in range(ly*lx):
        actrow = list(X[indice].copy())
        if actrow in known:
            new = rules(actrow)
            if new != -1:
                myy[indice] = new
    return myy


def nn2(X, Theta1, Theta2):
    Z2 = dot(hstack((X, ones([ly*lx, 1]))), transpose(Theta1))
    Z2 = sigmoid(Z2)
    y = dot(hstack((Z2, ones([ly*lx, 1]))), transpose(Theta2))
    y = argmax(y, axis=1)
    sol = applyrules(X, y)
    return sol.reshape(ly, lx)


def neighbors(state):
    a_entregar = list()
    for dy in range(ly):
        for dx in range(lx):
            a_entregar.append(
                [state[dy-1, dx-1], state[dy-1, dx], state[dy-1, (dx+1)%lx],
                state[dy, dx-1], state[dy, dx], state[dy, (dx+1)%lx],
                state[(dy+1)%ly, dx-1], state[(dy+1)%ly, dx], state[(dy+1)%ly, (dx+1)%lx]])
    return a_entregar


def myerror(thetas):
    theta1 = thetas[:(numbneig)*a].reshape(a,numbneig)
    theta2 = thetas[(numbneig)*a:(numbneig)*a+numbstates*(a+1)].reshape(numbstates, a+1)
    actualca = ca0.copy()
    serror = 0
    for t in range(t2):
        actualca = nn2(neighbors(actualca), theta1, theta2)
        if t == t1-1:
            serror += mysum(actualca != ds1)
        elif t == t2-1:
            serror += mysum(actualca != ds2)
    return serror


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
