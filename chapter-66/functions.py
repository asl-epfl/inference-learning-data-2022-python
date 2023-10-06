import numpy as np
from tqdm import tqdm

def auto_encoder(feature_data,t_hidden,t_output,n2,mu,rho,passes_ac):

    # Initialization
    L = 3 # total number of layers, including input and output layers --> ONE hidden layer
    n1 = feature_data.shape[-1] # size of input layer, which is equal to M
    nL = n1 # size of output layer, which is equal to the number of labels
    n3 = n1 # number of nodes in fictitious layer 3
    Q = nL # size of output layer; same as nL, which the number of classes as well.

    W = (1/np.sqrt(n1))*np.random.randn(n2, n1)
    theta = np.random.randn(n2)
    Wx = (1/np.sqrt(n2))*np.random.randn(n1, n2)
    thetax = np.random.randn(n1)

    yCell = [None]*L # to save the y vectors across layers
    zCell = [None]*L # to save the z vectors across layers
    dCell = [None]*L # to save the sensitivity delta vectors
    Wcell = [W, Wx] # a cell array containing the weight matrices of different dimensions
    ThetaCell = [theta, thetax] # a cell array for the thetas

    # Training using random reshuffling
    N = feature_data.shape[0] # numbe of data points

    for px in tqdm(range(passes_ac)):
        Px = np.random.permutation(N) # using random reshuffling
        for n in range(N): # training a neural network with one hidden layer
            h = feature_data[Px[n]] # a column vector
            gamma = h.copy()

            y = h.copy()
            yCell[0] = y.copy()

            # FORWARD PROPAGATION
            ell = 0 # first hidden layer
            Weight = Wcell[ell]
            theta = ThetaCell[ell]
            y = yCell[ell]
            z = Weight@y - theta
            zCell[ell+1] = z.copy() # save z_{ell+1}

            K = z.shape[0]
            y = np.zeros(K) # let us now generate y_{ell+1}; same size as z

            if t_hidden == 1: # sigmoid
                y = 1/(1+np.exp(-z))
            elif t_hidden == 2: # tanh
                a = np.exp(z) - np.exp(-z)
                b = np.exp(z) + np.exp(-z)
                y = a/b 
            elif t_hidden == 3: # rectifier
                y = np.array([max(0, z[k]) for k in range(K)])
            elif t_hidden == 4: # linear
                y = z.copy()
            yCell[ell+1] = y.copy() # save y_{ell+1}

            ell = 1 # output layer
            Weight = Wcell[ell]
            theta = ThetaCell[ell]
            y = yCell[ell]
            z = Weight@y - theta
            zCell[ell+1] = z.copy() # save z_{ell+1}

            K = z.shape[0]
            y = np.zeros(K) # let us now generate y_{ell+1}; same size as z

            if t_output == 1: # sigmoid
                y = 1/(1+np.exp(-z))
            elif t_output == 2: # tanh
                a = np.exp(z) - np.exp(-z)
                b = np.exp(z) + np.exp(-z)
                y = a/b 
            elif t_output == 3: # rectifier
                y = np.array([max(0, z[k]) for k in range(K)])
            elif t_output == 4: # linear
                y = z.copy()
            yCell[ell+1] = y.copy() # save y_{ell+1}

            zL = zCell[-1]
            yL = yCell[-1]
            K = zL.shape[0]
            gamma_hat = yL.copy()

            J = np.zeros((K, K))
            if t_output == 1: # sigmoid
                f = 1/(1+np.exp(-zL))
                J = np.diag(f*(1-f)) # computing f'(z_L) in diagonal matrix form
            elif t_output == 2: # tanh
                b = np.exp(zL) + np.exp(-zL) # computing f'(z_L) in diagonal matrix form
                J == np.diag(4/b**2)
            elif t_output == 3: # rectifier
                for k in range(K):
                    if z[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                        J[k, k] = 0
                    elif z[k] > 0:
                        J[k, k] = 1
                    elif z[k] < 0:
                        J[k, k] = 0
            elif t_output == 4: # linear
                J = np.eye(K)
            
            deltaL = 2*J@(gamma_hat - gamma)
            dCell[-1] = deltaL.copy() # boundary delta

            # BACKPROPAGATION
            ell = L - 1 # start the backward propagation
            Weight_before = Wcell[ell-1]
            theta_before = ThetaCell[ell-1]
            y = yCell[ell-1]
            delta = dCell[ell]

            Weight = (1-2*mu*rho)*Weight_before - mu*delta.reshape(-1, 1)@y.reshape(1, -1)
            Wcell[ell-1] = Weight.copy() # update weight

            theta = theta_before + mu*delta 
            ThetaCell[ell-1] = theta # update theta

            if ell >= 2: # computing next delta only for ell >= 2
                z = zCell[ell-1]
                K = z.shape[0]
                J = np.zeros((K, K))
                #we should use here the activation of the HIDDEN layer
                if t_hidden == 1: # sigmoid
                    f = 1/(1+np.exp(-z))
                    J = np.diag(f*(1-f))
                elif t_hidden == 2: # tanh
                    b = np.exp(z) + np.exp(-z)
                    J = np.diag(4/b**2)
                elif t_hidden == 3: # rectifier
                    for k in range(K):
                        if z[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                            J[k, k] = 0
                        elif z[k] > 0:
                            J[k, k] = 1
                        elif z[k] < 0:
                            J[k, k] = 0
                elif t_hidden == 4: # linear
                    J = np.eye(K)
                dCell[ell-1] = J@((Weight_before).T@delta)

            ell = 1 # next backward iteration
            Weight_before = Wcell[ell-1]
            theta_before = ThetaCell[ell-1]
            y = yCell[ell-1]
            delta = dCell[ell]

            Weight = (1-2*mu*rho)*Weight_before - mu*delta.reshape(-1, 1)@y.reshape(1, -1) 
            Wcell[ell-1] = Weight # update weight

            theta = theta_before + mu*delta
            ThetaCell[ell-1] = theta.copy() # update theta
    W = Wcell[0]
    theta = ThetaCell[0]
    return W, theta

def apply_activation(type, z):

    P = len(z)
    y = np.zeros(P)

    for p in range(P):
        if type == 1:  # sigmoid
            y[p] = 1 / (1 + np.exp(-z[p]))
        elif type == 2:  # tanh
            a = np.exp(z[p]) - np.exp(-z[p])
            b = np.exp(z[p]) + np.exp(-z[p])
            y[p] = a / b
        elif type == 3:  # rectifier
            y[p] = max(0, z[p])
        elif type == 4:  # linear
            y[p] = z[p]

    return y

def forward(H, W, theta, type):
    # H: NxM data matrix with rows as feature vectors
    # W: N2xM
    # theta: N2 x 1
    # type of activation function: 1=sigmoid, 2=tanh, 3=rectifier

    N = H.shape[0]
    N2 = W.shape[0]
    H2 = np.zeros((N, N2))

    for n in range(N):
        h = H[n, :]
        z = W@h - theta
        H2[n, :] = apply_activation(type, z)

    return H2

def vector_bernoulli(x, p):
    P = max(p.shape)
    prod = 1
    for m in range(P):
        a = p[m]**x[m]
        b = (1-p[m])**(1-x[m])
        prod *= a*b 

    return prod

def rand_bernoulli(y):
    # y: input vector with entries in the range [0,1];
    # yb: output vector with binary entries 0,1.
    # Prob(n-th entry in yb = 1) is n-th entry of y

    P = max(y.shape)
    yb = np.zeros(P)
    for p in range(P):
        a = np.random.rand() # flip a coin; random number within [0,1]
        if a <= y[p]:
            yb[p] = 1
    
    return yb

def rand_bernoulli_matrix(H):
    # H: input matrix with entries in the range [0,1]
    # Hb: output matrix with binary entries 0,1.
    # Prob(m,n entry in Hb = 1) is m,n entry of H

    N = H.shape[0]
    M = H.shape[1]
    Hb = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            a = np.random.rand() # % flip a coin; random number within [0,1]
            if a <= H[n, m]:
                Hb[n, m] = 1

    return Hb

def contrastive_divergence(feature_data,n2,mu,flag,passes_cd):

    # Initialization
    n1 = feature_data.shape[-1] # number of input nodes; also equal to size of feature vector

    W = np.random.randn(n2, n1)
    theta = np.random.randn(n2)
    theta_r = np.random.randn(n1)

    N = feature_data.shape[0] # number of data points
    for py in tqdm(range(passes_cd)):
        Py = np.random.permutation(N)
        for n in range(N):
            hb = feature_data[Py[n]] # column vector

            z = W@hb - theta # forward pass
            y = apply_activation(1, z) # type =1 ==> must use sigmoid function
            yb = rand_bernoulli(y)

            zprime = W.T@yb - theta_r # backward pass
            hprime = apply_activation(1, zprime)
            hbprime = rand_bernoulli(hprime)

            z3 = W@hbprime - theta # second forward pass 
            yprime = apply_activation(1, z3)
            ybprime = rand_bernoulli(yprime)

            if flag == 1: # binary features
                A = yb.reshape(-1, 1)@hb.reshape(1, -1) - ybprime.reshape(-1, 1)@hbprime.reshape(1, -1) 
                W += mu*A 
                theta += mu*(ybprime-yb)
                theta_r += mu*(hbprime-hb)
            else:
                A = yb.reshape(-1, 1)@hb.reshape(1, -1) - ybprime.reshape(-1, 1)@hbprime.reshape(1, -1) # when the feature vectors are NOT necessarily binary-valued
                W += mu*A 
                theta += mu*(ybprime-yb)
                theta_r += mu*(hprime-hb)

    return W, theta, theta_r

def nn_entropy_softmax(layers,act,nc,passes,mu,rho,features_train,labels_train,features_test,labels_test,Wcell_init,ThetaCell_init):

    # layers: number of layers, including input and output layers
    # act: type of activation function; 1=sigmoid, 2=tanh, 3=rectifier
    # nc: number of classes in the data
    # passes: number of passes over data
    # mu: step-size
    # rho: l2-regularization parameter

    # features_train: each row is Mx1 feature vector
    # labels_train: each entry is an integer label assuming nonnegative values: 0,1,2...
    # features_test: each row is Mx1 feature test vector
    # labels_test: each entry is an integer label assuming nonnegative values: 0,1,2...

    # Wcell_init: initial weight vectors for all layers in a cell structure
    # ThetaCell_init: initial bias vectors for all layers in a cell structure

    # The function returns:
    # Wcell: final weight vectors
    # ThetaCell: final bias vectors
    # error_test: number of errors over test data
    # error_train: number of errors over train data

    if act == 1:
        activation = 'sigmoid'
    elif act == 2:
        activation = 'tanh'
    elif act == 3:
        activation = 'rectifier'

    L = layers
    nsizes = np.zeros(L, dtype=int)
    nsizes[0] = features_train.shape[1]
    nsizes[-1] = nc
    Q = nsizes[-1]

    N_train = features_train.shape[0]
    M = features_train.shape[1]

    Wcell = Wcell_init.copy()
    ThetaCell = ThetaCell_init.copy()
    yCell = [None]*L  # to save the y vectors across layers
    zCell = [None]*L  # to save the z vectors across layers
    dCell = [None]*L  # to save the sensitivity delta vectors

    number_of_passes = passes 

    for p in range(number_of_passes):
        P = np.random.permutation(N_train) # using random reshuffling
        for n in tqdm(range(N_train)):
            h = features_train[P[n]] # a column vector
            m = labels_train[P[n]] # we are assuming labels are nonnegative integers: 0,1,2,..
            gamma = np.zeros(Q) # transform the class label into a column vector with all zeros and one location at one
            gamma[m] = 1

            y = h.copy()
            yCell[0] = y

            for ell in range(L-1): # forward propagation
                Weight = Wcell[ell]
                theta = ThetaCell[ell]
                y = yCell[ell]
                z = Weight@y - theta 
                zCell[ell+1] = z.copy() # save z_{ell+1}

                K = z.shape[0]
                y = np.zeros(K) # let us now generate y_{ell+1}; same size as z

                if activation == "sigmoid":
                    y = 1/(1+np.exp(-z))
                elif activation == "tanh":
                    a = np.exp(z) - np.exp(-z)
                    b = np.exp(z) + np.exp(-z)
                    y = a/b 
                elif activation == "rectifier":
                    for k in range(K):
                        y[k] = max(0, z[k])
                yCell[ell+1] = y.copy() # save y_{ell+1}

            zL = zCell[-1]
            yL = yCell[-1]
            K = zL.shape[0]

            gamma_hat = np.exp(zL)/np.exp(zL).sum() # softmax; output is softmax {z_{L}}
            # end of forward propagation

            # transition to backward propagation

            J = np.zeros((K, K)) # softmax is used
            deltaL = (gamma_hat - gamma) # no need for J here
            dCell[-1] = deltaL.copy() # boundary delta

            for ell in range(L-1, 0, -1): # start of backward propagation
                Weight_before = Wcell[ell-1]
                theta_before = ThetaCell[ell-1]
                y = yCell[ell-1]
                delta = dCell[ell]

                Weight = (1-2*mu*rho)*Weight_before - mu*delta.reshape(-1, 1)@y.reshape(1, -1)
                Wcell[ell-1] = Weight.copy() # update weight

                if ell >= 2: # computing next delta only for ell >= 2
                    z = zCell[ell-1]
                    K = z.shape[0]
                    J = np.zeros((K, K))
                    if activation == "sigmoid":
                        f = 1/(1+np.exp(-z))
                        f = f*(1-f)
                        J = np.diag(f)
                    elif activation == "tanh":
                        b = np.exp(z) + np.exp(-z)
                        J = np.diag(4/b**2)
                    elif activation == "rectifier":
                        for k in range(K):
                            if z[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                                J[k, k] = 0
                            elif z[k] > 0:
                                J[k, k] = 1
                            else:
                                J[k, k] = 0
                    dCell[ell-1] = J@((Weight_before).T@delta)

    yCell = [None]*L
    zCell = [None]*L 

    # Testing
    N_test = features_test.shape[0]
    test = features_test
    labels_hat = np.zeros(N_test) # used to save the predicted labels for comparison with labels_test
    output = np.zeros((Q, N_test))

    error_test = 0
    for n in range(N_test):
        h = test[n] # a column vector
        m = labels_test[n] # labels assumed to be nonnegative: 0,1,2,...
        gamma = np.zeros(Q) # transform the label into a vector with all zeros except at a single position with onme
        gamma[m] = 1 # a vector with unit entry

        yCell[0] = h.copy()
        for ell in range(L-1): # forward propagation
            Weight = Wcell[ell]
            theta = ThetaCell[ell]
            y = yCell[ell]
            z = Weight@y - theta

            K = z.shape[0]
            y = np.zeros(K) # generating next y

            if activation == "sigmoid":
                y = 1/(1+np.exp(-z))
            elif activation == "tanh":
                a = np.exp(z) - np.exp(-z)
                b = np.exp(z) + np.exp(-z)
                y = a/b 
            elif activation == "rectifier":
                for k in range(K):
                    y[k] = max(0, z[k])
            yCell[ell+1] = y.copy() # save y_{ell+1}

        zL = z.copy()
        yL = yCell[-1]

        K = zL.shape[0]

        gamma_hat = np.exp(zL)/np.exp(zL).sum() # softmax

        ax = np.max(gamma_hat) # find location of largest probability
        idx = np.argmax(gamma_hat) 
        labels_hat[n] = idx # location defines the label
        if labels_test[n] != labels_hat[n]:
            error_test += 1

    # Training 
    N_train = features_train.shape[0]
    train = features_train
    labels_hat = np.zeros(N_train) # used to save the predicted labels for comparison with labels_test
    output = np.zeros((Q, N_train))

    error_train = 0
    for n in range(N_train):
        h = train[n] # a column vector
        m = labels_train[n] # labels assumed to be nonnegative: 0,1,2,...
        gamma = np.zeros(Q) # transform the label into a vector with all zeros except at a single position with onme
        gamma[m] = 1 # a vector with unit entry

        yCell[0] = h.copy()
        for ell in range(L-1): # forward propagation
            Weight = Wcell[ell]
            theta = ThetaCell[ell]
            y = yCell[ell]
            z = Weight@y - theta

            K = z.shape[0]
            y = np.zeros(K) # generating next y

            if activation == "sigmoid":
                y = 1/(1+np.exp(-z))
            elif activation == "tanh":
                a = np.exp(z) - np.exp(-z)
                b = np.exp(z) + np.exp(-z)
                y = a/b 
            elif activation == "rectifier":
                for k in range(K):
                    y[k] = max(0, z[k])
            yCell[ell+1] = y.copy() # save y_{ell+1}

        zL = z.copy()
        yL = yCell[-1]

        K = zL.shape[0]

        gamma_hat = np.exp(zL)/np.exp(zL).sum() # softmax

        ax = np.max(gamma_hat) # find location of largest probability
        idx = np.argmax(gamma_hat) 
        labels_hat[n] = idx # location defines the label
        if labels_train[n] != labels_hat[n]:
            error_train += 1

    return Wcell, ThetaCell, error_test, error_train