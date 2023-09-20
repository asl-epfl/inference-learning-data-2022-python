import numpy as np
from tqdm import tqdm

def logistic_regression_2(training,testing,train_labels,test_labels,passes,mu,rho):

    N_test = max(testing.shape)
    N_train = max(training.shape)
    A_test = testing
    A_train = training
    C_test = test_labels
    C_train = train_labels
    M = min(training.shape)

    number_of_passes = passes # number of passes over data
    w = np.random.randn(M+1) # we extend w from two dimensions to three; its leading entry is interpreted as -theta
    Wmatrix = np.zeros((M+1, N_train))
    accuracy_curve = np.zeros(N_train)

    A_train = np.concatenate([np.ones((N_train, 1)), A_train], axis=-1)
    A_test = np.concatenate([np.ones((N_test, 1)), A_test], axis=-1)

    for p in range(number_of_passes):
        P = np.random.permutation(N_train)
        for n in range(N_train):
            h = A_train[P[n]] # feature vector (it is a row extended by adding one)
            gamma = C_train[P[n]] # its class
            gamma_hat = h@w 
            s = 1 + np.exp(gamma*gamma_hat)
            w = (1-mu*rho)*w + mu*gamma*h.T*(1/s)
            if p == 0:
                Wmatrix[:, n] = w # save the w' s from first pass to generate error curve
    
    wstar = w.copy()

    # computing likelihood that each test vector belongs to class +1
    test_predictions = np.zeros(N_test)
    likelihood = np.zeros(N_test)
    error = 0
    for n in range(N_test):
        h = A_test[n] # feature vector
        gamma = int(C_test[n][0]) # its class
        gamma_hat = h@wstar
        s = 1 + np.exp(-gamma*gamma_hat) # there is no gamma here because we want the likelihood of belonging to class +1
        likelihood[n] = 1/s
        if gamma*gamma_hat <= 0:
            error += 1
        if gamma_hat >= 0:
            test_predictions[n] = 1
        else:
            test_predictions[n] = -1

    E = (error/N_test)*100 # empirical error

    # error curve
    for nx in range(N_train):
        wx = Wmatrix[:, nx]
        error = 0
        for n in range(N_test):
            h = A_test[n] # feature vector
            gamma = C_test[n] # its class 
            gamma_hat = h@wx
            if gamma*gamma_hat <= 0:
                error += 1
        accuracy_curve[nx] = 1 - (error/N_test)
    
    return wstar,test_predictions,likelihood,E,accuracy_curve

def logistic_regression(training,testing,train_labels,test_labels,passes,mu,rho):

    N_test = max(testing.shape)
    N_train = max(training.shape)
    A_test = testing
    A_train = training
    C_test = test_labels
    C_train = train_labels
    M = min(training.shape)

    number_of_passes = passes # number of passes over data
    w = np.zeros(M+1) # we extend w from two dimensions to three; its leading entry is interpreted as theta

    A_train = np.concatenate([np.ones((N_train, 1)), A_train], axis=-1)
    A_test = np.concatenate([np.ones((N_test, 1)), A_test], axis=-1)

    for p in range(number_of_passes):
        P = np.random.permutation(N_train)
        for n in range(N_train):
            h = A_train[P[n]] # feature vector (it is a row and extended by adding one)
            gamma = C_train[P[n]] # its class
            gamma_hat = h@w 
            s = 1 + np.exp(gamma*gamma_hat)
            w = (1-mu*rho)*w + mu*gamma*h.T*(1/s)
    
    wstar = w.copy()

    # computing likelihood that each test vector belongs to class + 1
    error = 0
    test_predictions = np.zeros(N_test)
    likelihood = np.zeros(N_test)
    for n in range(N_test):
        h = A_test[n] # feature vector
        gamma = C_test[n] # its class
        gamma_hat = h@wstar 
        s = 1 + np.exp(-gamma*gamma_hat) # there is no gamma here because we want the likelihood og belonging to class + 1
        likelihood[n] = 1/s 
        if gamma*gamma_hat <= 0:
            error += 1
        if gamma_hat >= 0:
            test_predictions[n] = +1
        else:
            test_predictions[n] = -1
    
    E = (error/N_test)*100 # empirical error
    return wstar,test_predictions,likelihood,E

def generate_separable_logistic(N,wo,theta):

    M = max(wo.shape)
    gamma_vec = np.zeros(N) # gamma will end up with +1 and -1
    H = np.random.randn(M, N) # each column is a feature vector

    counter = 0
    for n in range(N):
        h = H[:, n] # feature vector
        px = 1/(1+np.exp(-(h.T@wo-theta))) # probability of gamma = +1
        if px >= 1/2:
            gamma_vec[n] = +1
            counter += 1
        else:
            gamma_vec[n] = -1
    
    p = counter/N # fraction of +1's in the result
    
    return p, gamma_vec, H

def logistic_minimizer(rho, gamma_vec, H, flag):
    M, N = H.shape
    mu = 0.2  # step-size
    
    w = np.zeros(M).reshape(-1,1)
    theta = 0
    
    for i in tqdm(range(5000)):
        s = np.zeros(M).reshape(-1,1)
        s2 = 0
        for n in range(N):
            h_n = H[:, n].reshape(-1,1)  # feature vector
            gamma_n = gamma_vec[n]  # label
            a = np.exp(-gamma_n * (np.dot(h_n.T, w) - theta))
            s += gamma_n * h_n * a / (1 + a)  # gradient vector for w
            s2 += flag * gamma_n * a / (1 + a)  # gradient vector for theta
        grad = s / N
        grad2 = s2 / N
        
        w = w - mu * (2 * rho) * w + mu * grad
        theta = theta - flag * mu * gamma_n * grad2  # flag=0 ==> theta stays at 0
        
    ws = w.copy()
    thetas = theta
    
    # Calculate minimum risk value, denoted by Ps.
    Ps = 0
    for m in range(N):
        h_m = H[:, m].reshape(-1,1)  # feature vector
        gamma_m = gamma_vec[m]  # label
        a = 1 + np.exp(-gamma_m * (np.dot(h_m.T, ws) - thetas))
        Ps += np.log(a)
    Ps = Ps / N
    Ps = Ps + rho * np.linalg.norm(ws, 2)**2
    
    # Testing the optimality of ws and thetas
    s = np.zeros(M).reshape(-1,1)
    s2 = 0
    for n in range(N):
        h_n = H[:, n].reshape(-1,1)  # feature vector
        gamma_n = gamma_vec[n]  # label
        a = np.exp(-gamma_n * (np.dot(h_n.T, ws) - thetas))
        s += gamma_n * h_n * a / (1 + a)  # gradient vector
        s2 += flag * gamma_n * a / (1 + a)
    grad = s / N
    np.linalg.norm(2*rho*ws - grad) # ws is optimal if this variable is ~ 0
    
    if flag == 1:
        grad2 = s2 / N # thetas is optimal is this variable is ~ 0
    
    return ws, Ps, thetas