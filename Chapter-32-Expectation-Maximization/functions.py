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

def vector_bernoulli(x, p):
    P = max(x.shape)
    prod = 1
    for m in range(P):
        a = p[m]**x[m]
        b = (1-p[m])**(1-x[m])
        prod = prod*a*b

    return prod

def EM_GMM(input_vector, K, M):

    input_vector = input_vector[:, :]
    [P, N] = input_vector.shape
    
    mu_hat = np.zeros((P, K))
    R_hat = np.zeros((P, P, K))
    pi_hat = np.zeros(K)

    D = np.diag(input_vector.T.std(axis=0)) #estimates the standard deviation of each entry in input vector
    R_init = D@D #uses the square of the estimates to define an initial diagonal matrix

    E = input_vector.T.mean(axis=0).reshape(1, -1) #estimate the mean of each entry in input vector
    mean_init = E.T

    for k in range(K): #initializing Px1 mean vectors, PxP covariance matrices, and priors for K components
        mu_hat[:, k] = (mean_init + 2*np.random.rand(P, 1)).reshape(P) #adds randomness across the components
        R_hat[:, :, k] = R_init + (np.random.randint(P)+1)*np.eye(P) #adds randomness across the components
        pi_hat[k] = 1/K
    
    N_hat = np.zeros(K)
    r_hat = np.zeros((K, N)) #each column indicates the likelihood for n-th input vector to belong to components

    #Running EM
    for m in range(M):
        print(m)
        #E-step
        for k in range(K):
            mu_hat_k = mu_hat[:, k]
            R_hat_k = R_hat[:, :, k]
            pi_hat_k = pi_hat[k]

            for n in range(N):
                y = input_vector[:, n]
                a = pi_hat_k*vector_gaussian(y, mu_hat_k.reshape(1, -1), R_hat_k)
                d = 0
                for j in range(K):
                    d += pi_hat[j]*vector_gaussian(y, mu_hat[:, j].reshape(1, -1), R_hat[:, :, j])
                
                r_hat[k, n] = a/d
            N_hat[k] = r_hat[k, :].sum()    
        
        #M-step
        z_vec = np.zeros((P, K))
        for k in range(K):
            z_vec[:, k] = r_hat[k]@input_vector.T
            mu_hat[:, k] = z_vec[:, k] / N_hat[k]
        
        S = np.zeros((P, P, K))
        for k in range(K):
            for n in range(N):
                y = input_vector[:, n]
                yc = y - mu_hat[:, k]
                S[:, :, k] += r_hat[k, n]*(yc.reshape(1, -1).T@yc.reshape(1, -1))
            R_hat[:, :, k] = S[:, :, k] / N_hat[k]
            pi_hat[k] = N_hat[k]/N

            episilon = 0.00001
            R_hat[:, :, k] += episilon*np.eye(P) #add small pertubation to avoid singular R's


    
    ind = r_hat.argmax(axis=0)
    lkd = r_hat.max(axis=0)

    return pi_hat, lkd, ind, r_hat, mu_hat, R_hat

     
    