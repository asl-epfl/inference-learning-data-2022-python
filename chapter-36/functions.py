import numpy as np

def scalar_gaussian(x, mean, variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-(x-mean)**2/(2*variance))

def dirichlet(alpha):
    B = max(alpha.shape)
    x = np.zeros(B)
    for b in range(B):
        x[b] = np.random.gamma(shape=alpha[b], scale=1) # generates Gamma-distributed RV
    return x/sum(x) # normalize by the sum to get Dirichlet

def beta(lambda1, lambda2):
    B = 2
    alpha = [lambda1, lambda2]
    x = np.zeros(B)
    for b in range(B):
        x[b] = np.random.gamma(alpha[b], 1) # generates Gamma-distributed RV
    x = x/sum(x)
    return x[0]

def mode_beta(a, b):
    flag = 0 # one mode only

    if (a > 1) and (b > 1):
        x = (a-1)/(a+b-2)

    elif (a < 1) and (b < 1): # two modes ate= 0 and at 1, return one of them at random
        x = np.random.randint(0, 2)

    elif (a <= 1) and (b > 1):
        x = 0
    
    elif (a > 1) and (b <= 1):
        x = 1
    
    else:
        x = np.random.rand() # any number between 0 and 1
        if x == 0:
            x += 0.000001
        elif x == 1:
            x -= 0.000001
    
    return x