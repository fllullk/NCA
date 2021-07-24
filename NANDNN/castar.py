from numpy import array


namebestsolfile = "thetasol.txt"
nameant = "thetas.txt"

lb = -5
ub = 5

itersopt = 10

alfa = 0.001

numbneig = 2 + 1 # +1 due to bias
numbstates = 2 #number of state, 0 or 1
numbvars = (numbstates)*(numbneig)

x0 = array([[0, 0], [0, 1], [1, 0], [1, 1]])

ndata = len(x0)

ds = array([1, 1, 1, 0])
