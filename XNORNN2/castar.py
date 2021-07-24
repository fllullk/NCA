from numpy import array


namebestsolfile = "thetasol.txt"
nameant = "thetas.txt"

lb = -5
ub = 5

itersopt = 50

alfa = 0.001

numbneig = 2 + 1 # +1 due to bias
a = 2
numbstates = 2 #number of state, 0 or 1
numbvars = a*(numbneig) + numbstates*(a+1)

x0 = array([[0, 0], [0, 1], [1, 0], [1, 1]])

ndata = len(x0)

ds = array([1, 0, 0, 1])
