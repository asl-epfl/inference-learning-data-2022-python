import numpy as np
import scipy
from scipy.special import gamma

def gamma_derivative(z):
    a = 0.5772156649
    N = 1000
    s = -a
    for k in range(N):
        s = s - (1/(z+k)) + (1/(k+1))
    return s