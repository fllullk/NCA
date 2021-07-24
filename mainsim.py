from numpy import (array, transpose, dot, exp, argmax, linspace, ones, hstack,
                   rot90, fliplr)
from numpy import sum as mysum
from matplotlib.pyplot import imshow, show, subplots
from matplotlib.animation import FuncAnimation
from castar import (lx, ly, ca0, numbneig, a, numbstates, known)


myiterations = 10

print(ca0)

imshow(ca0 ,cmap="binary")
show()

thetas2 = []


with open("thetas.txt", "r") as file:
    for line in file.readlines():
        thetas2.append(float(line.strip()))


thetas2 = array(thetas2)
mytheta1 = thetas2[:(numbneig)*a].reshape(a,numbneig)
mytheta2 = thetas2[(numbneig)*a:(numbneig)*a+numbstates*(a+1)].reshape(numbstates, a+1)    


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


def simnext(state, Theta1, Theta2):
    return nn2(neighbors(state), Theta1, Theta2)


# print(nn2(neighbors(ca0), mytheta1, mytheta2))

imshow(simnext(ca0, mytheta1, mytheta2) ,cmap="binary")
show()

ca = ca0.copy()

fig, ax = subplots()

ln = imshow(ca, vmin = 0, vmax = numbstates-1, cmap ="binary")

def init():
    ax.set_xlim(0, lx)
    ax.set_ylim(ly, 0)
    return ln,

def update(frame):
    global ca
    ca = simnext(ca, mytheta1, mytheta2)
    ln = imshow(ca, vmin = 0, vmax = numbstates-1, cmap ="binary")
    return ln,

ani = FuncAnimation(fig, update, frames=linspace(0, myiterations - 1, myiterations), init_func=init, blit=True)
show()
