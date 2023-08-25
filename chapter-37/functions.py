import numpy as np

def dirichlet(alpha):
    B = max(alpha.shape)
    x = np.zeros(B)
    for b in range(B):
        x[b] = np.random.gamma(shape=alpha[b], scale=1) # generates Gamma-distributed RV
    return x/sum(x) # normalize by the sum to get Dirichlet

def trigamma(z):

    if z == 0:
        y = np.NaN # evaluates to infinity at z = 0
    else:
        N = 1000
        s = 0
        for n in range(N):
            s = s -1/((z+n)**2)
    return s

def categorical_variable(theta):

    B = max(theta.shape)
    cum_p = np.zeros(B) # cummulative density
    for b in range(B):
        cum_p[b] = sum(theta[:b+1])
    
    idx = 1
    x = np.random.rand()
    for b in range(B):
        if x > cum_p[b]: #categorical distribution
            idx += 1
    
    return idx