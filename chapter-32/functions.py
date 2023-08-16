import numpy as np

def scalar_gaussian(x, mean, variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-(x-mean)**2/(2*variance))

def vector_gaussian(x, mean, R):
    P = max(x.shape)
    a = (2*np.pi)**(P/2)
    a = 1/a
    b = np.sqrt(np.linalg.det(R))
    b = 1/b

    c = ((x.reshape(1, -1)-mean)@np.linalg.inv(R)@(x.reshape(1, -1)-mean).T)[0][0]
    c = -c/2
    return a*b*np.exp(c)
    