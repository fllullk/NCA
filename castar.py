from numpy import zeros, int8, array, rot90, fliplr


namebestsolfile = "thetasol.txt"
nameant = "thetas.txt"
lx = 12
ly = 12
lb = -10
ub = -lb
itersopt = 10

alfa = 0.001

numbneig = 9 + 1 # +1 due to bias node
a = 15
numbstates = 3 #number of state, 0 or 1
numbvars = a*(numbneig) + numbstates*(a + 1)

ca0 = zeros([ly,lx], dtype=int8)

#ca0[1, 5] = 2
ca0[2, 4:6+1] = 2
ca0[3, 4] = 2
ca0[3, 6] = 2
ca0[4, 2:4+1] = 2
ca0[4, 6:8+1] = 2
ca0[5, 1:2+1] = 2
ca0[5, 8:9+1] = 2
ca0[6, 2:4+1] = 2
ca0[6, 6:8+1] = 2
ca0[7, 4] = 2
ca0[7, 6] = 2
ca0[8, 4:6+1] = 2
ca0[9, 5] = 2

ca0[3:7+1, 5] = 1
ca0[5, 3:7+1] = 1

known = [
    list(array([0,0,0,0,0,0,0,0,0])),

    list(array([0,0,0,0,2,0,2,2,2])),
    list(array(rot90(array([0,0,0,0,2,0,2,2,2]).reshape(3,3)).flat)),
    list(array(rot90(rot90(array([0,0,0,0,2,0,2,2,2]).reshape(3,3))).flat)),
    list(array(rot90(rot90(rot90(array([0,0,0,0,2,0,2,2,2]).reshape(3,3)))).flat)),

    list(array([0,0,2,0,2,2,0,2,1])),
    list(array(rot90(array([0,0,2,0,2,2,0,2,1]).reshape(3,3)).flat)),
    list(array(rot90(rot90(array([0,0,2,0,2,2,0,2,1]).reshape(3,3))).flat)),
    list(array(rot90(rot90(rot90(array([0,0,2,0,2,2,0,2,1]).reshape(3,3)))).flat)),
    list(array(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3)).flat)),
    list(array(rot90(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3))).flat)),
    list(array(rot90(rot90(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3)))).flat)),
    list(array(rot90(rot90(rot90(fliplr(array([0,0,2,0,2,2,0,2,1]).reshape(3, 3))))).flat)),

    list(array([0, 2, 0, 2, 2, 2, 2, 1, 2])),
    list(array(rot90(array([0, 2, 0, 2, 2, 2, 2, 1, 2]).reshape(3,3)).flat)),
    list(array(rot90(rot90(array([0, 2, 0, 2, 2, 2, 2, 1, 2]).reshape(3,3))).flat)),
    list(array(rot90(rot90(rot90(array([0, 2, 0, 2, 2, 2, 2, 1, 2]).reshape(3,3)))).flat)),

    list(array([0,2,2,0,2,1,2,2,1])),
    list(array(rot90(array([0,2,2,0,2,1,2,2,1]).reshape(3,3)).flat)),
    list(array(rot90(rot90(array([0,2,2,0,2,1,2,2,1]).reshape(3,3))).flat)),
    list(array(rot90(rot90(rot90(array([0,2,2,0,2,1,2,2,1]).reshape(3,3)))).flat)),
    list(array(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3)).flat)),
    list(array(rot90(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3))).flat)),
    list(array(rot90(rot90(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3)))).flat)),
    list(array(rot90(rot90(rot90(fliplr(array([0,2,2,0,2,1,2,2,1]).reshape(3, 3))))).flat)),

    list(array([2, 2, 2, 2, 1, 2, 2, 1, 2])),
    list(array(rot90(array([2, 2, 2, 2, 1, 2, 2, 1, 2]).reshape(3,3)).flat)),
    list(array(rot90(rot90(array([2, 2, 2, 2, 1, 2, 2, 1, 2]).reshape(3,3))).flat)),
    list(array(rot90(rot90(rot90(array([2, 2, 2, 2, 1, 2, 2, 1, 2]).reshape(3,3)))).flat)),

    list(array([0,2,1,2,2,1,1,1,1])),
    list(array(rot90(array([0,2,1,2,2,1,1,1,1]).reshape(3,3)).flat)),
    list(array(rot90(rot90(array([0,2,1,2,2,1,1,1,1]).reshape(3,3))).flat)),
    list(array(rot90(rot90(rot90(array([0,2,1,2,2,1,1,1,1]).reshape(3,3)))).flat)),
    list(array(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3)).flat)),
    list(array(rot90(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3))).flat)),
    list(array(rot90(rot90(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3)))).flat)),
    list(array(rot90(rot90(rot90(fliplr(array([0,2,1,2,2,1,1,1,1]).reshape(3, 3))))).flat)),

    list(array([2, 1, 2, 2, 1, 2, 1, 1, 1])),
    list(array(rot90(array([2, 1, 2, 2, 1, 2, 1, 1, 1]).reshape(3,3)).flat)),
    list(array(rot90(rot90(array([2, 1, 2, 2, 1, 2, 1, 1, 1]).reshape(3,3))).flat)),
    list(array(rot90(rot90(rot90(array([2, 1, 2, 2, 1, 2, 1, 1, 1]).reshape(3,3)))).flat)),

    list(array([2, 1, 2, 1, 1, 1, 2, 1, 2]))

]
