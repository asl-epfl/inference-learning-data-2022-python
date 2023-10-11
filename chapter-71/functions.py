import numpy as np
from tqdm import tqdm

def bernoulli_vector(p, L):
    # p: probability of zero entry
    # a: output vector with binary entries 0, 1
    # L: size of a 

    a = np.ones(L)
    for n in range(L):
        x = np.random.rand() # flip a count; random a number within [0, 1]
        if x <= p:
            a[n] = 0
    return a

def train_feedforward_neural_network(
        L, number_of_classes, mu, rho, number_of_passes, features_train, labels_train,
        n2, n3, dropout, activation, softmax, p_vec, cost):
    # Setting up the layers
    L = 4 # total number of layers, including input and output layers --> L-2 hidden layers
    nL = number_of_classes # size of output layer, which is equal to the number of labels
    n1 = features_train.shape[-1] # size of input layer, which is equal to M
    n4 = nL # same as output
    Q = nL # size of output layer; same as nL, which the number of classes as well.

    # Initialization
    W1 = (1/np.sqrt(n1))*np.random.randn(n2, n1)
    W2 = (1/np.sqrt(n2))*np.random.randn(n3, n2)
    W3 = (1/np.sqrt(n3))*np.random.randn(n4, n3)

    theta1 = np.random.randn(n2)
    theta2 = np.random.randn(n3)
    theta3 = np.random.randn(n4)

    Wcell = [W1, W2, W3] # a cell array containing the weight matrices of different dimensions
    ThetaCell = [theta1, theta2, theta3] # a cell array for the thetas

    Wcell_before = Wcell.copy()
    ThetaCell_before = ThetaCell.copy()

    # Training using random reshuffling
    N_train = features_train.shape[0]
    M = features_train.shape[1]

    yCell = [None]*L # to save the y vectors across layers
    zCell = [None]*L # to save the z vectors across layers
    dCell = [None]*L # o save the sensitivity delta vectors

    labels_train = labels_train.reshape(-1)

    for p in range(number_of_passes):
        P = np.random.permutation(N_train) # using random reshuffling
        for n in tqdm(range(N_train)):
            h = features_train[P[n], :] # a column vector
            m = labels_train[P[n]] # we are ussuming labels are nonnegative integers: 0, 1, 2, ...
            gamma = np.zeros(Q) # transform the class into a columns vector with all zeros and one location at one
            gamma[m] = 1

            y = h.copy()
            yCell[0] =y

            if dropout == 1:
                a_input = bernoulli_vector(p_vec[0], n1) # Bernoulli vector for input layer
                a_2 = bernoulli_vector(p_vec[1], n2) # Bernoulli vector for hidden layer 1
                a_3 = bernoulli_vector(p_vec[2], n3) # Bernoulli vector for hidden layer 2
                aCell = [a_input, a_2, a_3] # Bernoulli vectors
            
            for ell in range(L-1): # forward propagation
                Weight = Wcell[ell]
                theta = ThetaCell[ell]
                y = yCell[ell]

                if dropout == 0:
                    z = Weight@y - theta 
                else:
                    ab = aCell[ell]
                    yab = np.multiply(y, ab) # when dropout is used
                    z = Weight@yab - theta 
                
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
            yL = zCell[-1]
            K = zL.shape[0]
            gamma_hat = np.zeros(Q) # Q and K are equal

            if softmax == 0: # no softmax
                gamma_hat = yL.copy() # output is y_{L}
            else:
                gamma_hat = np.exp(zL)/np.exp(zL).sum() # softmax; output is softmax(z_{L})
            # end of forward propagation

            # transition to backward propagation
            J = np.zeros((K, K))
            if softmax == 0: # if no softmax is being used
                if activation == "sigmoid":
                    for k in range(K):
                        f = 1/(1+np.exp(-zL[k]))
                        J[k, k] = f*(1-f) # computing f'(z_L) in diagonal matrix form
                elif activation == "tanh":
                    for k in range(K):
                        b = np.exp(zL[k]) + np.exp(-zL[k]) # computing f'(z_L) in diagonal matrix form
                        J[k, k] = 4/(b**2)
                elif activation == "rectifier":
                    for k in range(K):
                        if zL[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                            J[k, k] = 0
                        elif zL[k] > 0:
                            J[k, k] = 1
                        elif zL[k] < 0:
                            J[k, k] = 0
            else: # softmax is used
                J = np.diag(gamma_hat) - gamma_hat@gamma_hat.T # his J is only used when cost=least-squares + softmax
            
            if cost == 1: # cross entropy + softmax are used together
                deltaL = (gamma_hat - gamma) # no need  for J here
            else:
                deltaL = 2*J@(gamma_hat - gamma)
            
            dCell[-1] = deltaL # boundary delta

            for ell in range(L-1, 0, -1): # start of backward propagation
                Weight_before = Wcell[ell-1]
                theta_before = ThetaCell[ell-1]
                y = yCell[ell-1]
                delta = dCell[ell]

                if dropout == 1: # adjusting Weight and theta when dropout is used
                    ab = aCell[ell-1]
                    yab = np.multiply(y, ab) # when dropout is used
                    Weight = (1-2*mu*rho)*Weight_before - mu*delta.reshape(-1, 1)@yab.reshape(1, -1)
                    theta = theta_before + mu*delta 
                
                Wcell[ell-1] = Weight.copy()
                ThetaCell[ell-1] = theta.copy() # update theta

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
                    if dropout == 0:
                        dCell[ell-1] = J@((Weight_before).T@delta)
                    else:
                        ab = aCell[ell-1]
                        dx = J@(Weight_before.T@delta)
                        dCell[ell-1] = np.multiply(dx, ab)

    if dropout == 1: # scalling the parameters at end of dropout step
        for ell in range(L-1, 0, -1):
            Weight = Wcell[ell-1]
            Wcell[ell-1] = (1-p_vec[ell-1])*Weight

            theta = ThetaCell[ell-1]
            ThetaCell[ell-1] = (1-p_vec[ell-1])*theta 

    return Wcell, ThetaCell

def get_inference_error(
        L, features_test, labels_test, number_of_classes, Wcell, ThetaCell,
        softmax, activation):
    yCell = [None]*L 
    zCell = [None]*L 

    Q = number_of_classes

    # Testing
    N_test = features_test.shape[0]
    test = features_test 
    labels_hat = np.zeros(N_test) # used to save the predicted labels for comparison with labels_test
    output = np.zeros((Q, N_test))

    labels_test = labels_test.reshape(-1) 

    error = 0
    for n in tqdm(range(N_test)):
        h = test[n, :].T # a column vector
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
        gamma_hat = np.zeros(K) # K and Q are equal to each other

        if softmax == 0:
            gamma_hat = yL.copy() # if no softmax is used
        else:
            gamma_hat = np.exp(zL)/np.exp(zL).sum() # when softmax is used
        
        if softmax == 1: # using softmax
            ax = np.max(gamma_hat) # find location of largest probability
            idx = np.argmax(gamma_hat) 
            labels_hat[n] = idx # location defines the label
            if labels_test[n] != labels_hat[n]:
                error += 1
        else: # no softmax is used; compare each output entry to 1/2
            for k in range(K):
                if activation == "sigmoid":
                    if gamma_hat[k] >= 1/2:
                        output[k, n] = 1
                        labels_hat[n] = k
                    else:
                        output[k, n] = 0 # output vector will have ones and zeros at active attributes
                elif activation == "tanh":
                    if gamma_hat[k] >= 0:
                        output[k, n] = 1
                        labels_hat[n] = k
                    else:
                        output[k, n] = 0 # output vector will have ones and zeros at active attributes
            if labels_test[n] != labels_hat[n]:
                error += 1

    return error

def train_feedforward_neural_network_with_batch_norm(
        L, number_of_classes, lambda_, epsilon, mu, rho, features_train, labels_train,
        n2, n3, B, activation, softmax, cost):

    nL = number_of_classes # size of output layer, which is equal to the number of labels
    n1 = features_train.shape[-1] # size of input layer, which is equal to M
    n4 = nL # same as output layer
    Q = nL # size of output layer; same as nL, which the number of classes as well.

    # Initialization
    W1 = (1/np.sqrt(n1))*np.random.randn(n2, n1)
    W2 = (1/np.sqrt(n2))*np.random.randn(n3, n2)
    W3 = (1/np.sqrt(n3))*np.random.randn(n4, n3)

    theta1 = np.random.randn(n2)
    theta2 = np.random.randn(n3)
    theta3 = np.random.randn(n4)

    Wcell = [W1, W2, W3] # a cell array containing the weight matrices of different dimensions
    ThetaCell = [theta1, theta2, theta3] # a cell array for the thetas

    Wcell_before = Wcell.copy()
    ThetaCell_before = ThetaCell.copy()

    # Training using random reshuffling
    N_train = features_train.shape[0]
    M = features_train.shape[1]

    yCell = [[[] for _ in range(B)] for _ in range(L)] # to save the y vectors across layers
    zCell = [[[] for _ in range(B)] for _ in range(L)] # to save the z vectors across layers
    dCell = [[[] for _ in range(B)] for _ in range(L)] # o save the sensitivity delta vectors
    aCell = [None]*(L-1) # to save the a scaling vectors
    zbarCell = [None]*L # to save the z_bar mean vectors
    zbarCellsmooth = [None]*L 
    SCell = [None]*L # to save the Sigma^2 variance matrices
    SCellsmooth = [None]*L
    gammaCell = [None]*B
    gammahatcell = [None]*B 
    DCell = [[[] for _ in range(B)] for _ in range(L-1)]
    zprimeCell = [[[] for _ in range(B)] for _ in range(L)]
    vCell = [[[] for _ in range(B)] for _ in range(L)]

    zbarCell[0] = np.zeros(n1)
    zbarCell[1] = np.zeros(n2)
    zbarCell[2] = np.zeros(n3)
    zbarCell[3] = np.zeros(n4)
    zbarCellsmooth[0] = np.zeros(n1)
    zbarCellsmooth[1] = np.zeros(n2)
    zbarCellsmooth[2] = np.zeros(n3)
    zbarCellsmooth[3] = np.zeros(n4)

    SCellsmooth[0] = np.zeros((n1, n1))
    SCellsmooth[1] = np.zeros((n2, n2))
    SCellsmooth[2] = np.zeros((n3, n3))
    SCellsmooth[3] = np.zeros((n4, n4))

    aCell[0] = np.zeros(n2)
    aCell[1] = np.zeros(n3)
    aCell[2] = np.zeros(n4)

    nsize = [n1, n2, n3, n4]

    number_of_passes = 1

    labels_train = labels_train.reshape(-1)

    for p in range(number_of_passes):
        P = np.random.permutation(N_train) # using random reshuffling 
        SCell[0] = epsilon*np.eye(n1)
        SCell[1] = epsilon*np.eye(n2)
        SCell[2] = epsilon*np.eye(n3)
        SCell[3] = epsilon*np.eye(n4)

        for n in tqdm(range(N_train)):
            U = [np.random.randint(N_train) for _ in range(B)] # select B random samples
            for b in range(B): # these are the input features for the batch
                yCell[0][b] = features_train[U[b]]
                m = labels_train[U[b]]
                gamma = np.zeros(Q) # one-hot encoding
                gamma[m] = 1
                gammaCell[b] = gamma
            
            for ell in range(L-1): # forward propagation; computation of z's from batch
                Weight = Wcell[ell]
                for b in range(B):
                    y = yCell[ell][b]
                    z = Weight@y 
                    zCell[ell+1][b] = z.copy() # save z_{ell+1}
                    zbarCell[ell+1] = zbarCell[ell+1] + z 
                zbarCell[ell+1] = zbarCell[ell+1]/B # mean vector z_bar

                for b in range(B):
                    x = zCell[ell+1][b] - zbarCell[ell+1]
                    SCell[ell+1] = SCell[ell+1] + (1/(B-1))*np.diag(np.diag(x.reshape(-1, 1)@x.reshape(1, -1))) # covariance S 
                
                # smoothing
                zbarCellsmooth[ell+1] = lambda_*zbarCellsmooth[ell+1] + (1-lambda_)*zbarCell[ell+1]
                SCellsmooth[ell+1] = lambda_*SCellsmooth[ell+1] + (1-lambda_)*SCell[ell+1]

                for b in range(B):
                    S = SCell[ell+1]
                    z = zCell[ell+1][b]
                    zbar = zbarCell[ell+1]
                    avec = aCell[ell]
                    A = np.diag(avec)
                    theta = ThetaCell[ell]

                    zx = np.linalg.inv(S)@(z-zbar)
                    v = A@zx - theta
                    vCell[ell+1][b] = v
                    zprimeCell[ell+1][b] = zx 

                    K = v.shape[0]
                    y = np.zeros(K) # let us now generate y_{ell+1}; same size as z

                    if activation == "sigmoid":
                        y = 1/(1+np.exp(-v))
                    elif activation == "tanh":
                        a = np.exp(v) - np.exp(-v)
                        b = np.exp(v) + np.exp(-v)
                        y = a/b 
                    elif activation == "rectifier":
                        for k in range(K):
                            y[k] = max(0, v[k])
                    yCell[ell+1][b] = y.copy() # save y_{ell+1}
            
            for b in range(B):
                vL = vCell[-1][b]
                yL = yCell[-1][b]
                K = vL.shape[0]
                gamma_hat = np.zeros(Q) # Q and K are equal

                if softmax == 0: # no softmax
                    gamma_hat = yL # output is y_{L}
                else:
                    gamma_hat = np.exp(vL)/np.exp(vL).sum() # softmax; output is softmax(v_{L})
                gammahatcell[b] = gamma_hat.copy()
                gamma = gammaCell[b]
                J = np.zeros((K, K))
                if softmax == 0: # if no softmax is being used
                    if activation == "sigmoid":
                        for k in range(K):
                            f = 1/(1+np.exp(-vL[k]))
                            J[k, k] = f*(1-f) # computing f'(z_L) in diagonal matrix form
                    elif activation == "tanh":
                        for k in range(K):
                            b = np.exp(vL[k]) + np.exp(-vL[k]) # computing f'(z_L) in diagonal matrix form
                            J[k, k] = 4/(b**2)
                    elif activation == "rectifier":
                        for k in range(K):
                            if vL[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                                J[k, k] = 0
                            elif vL[k] > 0:
                                J[k, k] = 1
                            elif vL[k] < 0:
                                J[k, k] = 0
                else: # softmax is used
                    J = np.diag(gamma_hat) - gamma_hat@gamma_hat.T # his J is only used when cost=least-squares + softmax
                
                if cost == 1: # cross entropy + softmax are used together
                    deltaL = (gamma_hat - gamma) # no need for J here
                else:
                    deltaL = 2*J@(gamma_hat - gamma)
                dCell[-1][b] = deltaL # boundary delta
            
            for ell in range(L-1, 0, -1): # start of backward propagation
                Weight_before = Wcell[ell-1]
                theta_before = ThetaCell[ell-1]
                avec_before = aCell[ell-1]
                nsizeC = nsize[ell]
                C = np.zeros((B, B, nsizeC))
                S = SCell[ell]

                for b in range(B):
                    z = zCell[ell][b]
                    zbar = zbarCell[ell]
                    for bprime in range(B):
                        zprime = zCell[ell][bprime]
                        zeta = np.zeros(nsizeC)
                        for j in range(nsizeC):
                            zeta[j] = 1/(np.sqrt(S[j, j]))
                            if b == bprime:
                                aux = 1-((zeta[j]*zeta[j]*(z[j]-zbar[j]))**2)/(B-1)
                                C[b, bprime, j] = ((B-1)/B)*zeta[j]*aux
                            else:
                                aux = 1+(zeta[j]*zeta[j]*(z[j]-zbar[j])*(zprime[j]-zbar[j]))
                                C[b, bprime, j] = -zeta[j]*aux/B 
                
                for b in range(B):
                    size_a = max(avec_before.shape)
                    Dx = np.zeros((size_a, size_a))
                    for i in range(size_a):
                        Dx[i, i] = avec_before[i]*C[b, b, i]
                    DCell[ell-1][b] = Dx
                
                na, nb = Weight_before.shape # na is n_{ell+1}, nb is n_{\ell} in our notation
                X = (1-2*mu*rho)*Weight_before

                Weight = np.zeros(X.shape)

                for i in range(nb): # i varies between 1 and n_{\ell} in our notation
                    for j in range(na): # j varies between 1 and n_{\ell+1} in our notation
                        sumb = 0
                        for b in range(B):
                            y = yCell[ell-1][b]
                            delta = dCell[ell][b]
                            sumx = 0
                            for bprime in range(B):
                                sumx += C[b, bprime, j]*y[i]
                            sumb += delta[j]*avec_before[j]*sumx 
                        sumb = sumb*mu/B 
                        Weight[j, i] = X[j, i] - sumb # using (j,i) because of our convention
                
                sumd = np.zeros(nsize[ell])
                for b in range(B):
                    delta = dCell[ell][b]
                    sumd += delta 
                sumd = sumd*mu/B 
                theta = theta_before + sumd 

                suma = np.zeros((size_a, size_a))
                for b in range(B):
                    delta = dCell[ell][b]
                    zprime = zprimeCell[ell][b]
                    suma += delta.reshape(-1, 1)+zprime.reshape(1, -1) 
                suma = suma*mu/B 
                avec = avec_before - np.diag(suma)

                Wcell[ell-1] = Weight.copy()
                ThetaCell[ell-1] = theta.copy() # update theta
                aCell[ell-1] = avec.copy()

                for b in range(B):
                    delta = dCell[ell][b]
                    if ell >= 2: # computing next delta only for ell >= 2
                        v = vCell[ell-1][b]
                        K = v.shape[0]
                        J = np.zeros((K, K))
                        if activation == "sigmoid":
                            f = 1/(1+np.exp(-v))
                            f = f*(1-f)
                            J = np.diag(f)
                        elif activation == "tanh":
                            b = np.exp(v) + np.exp(-v)
                            J = np.diag(4/b**2)
                        elif activation == "rectifier":
                            for k in range(K):
                                if v[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                                    J[k, k] = 0
                                elif v[k] > 0:
                                    J[k, k] = 1
                                else:
                                    J[k, k] = 0
                        Dx = DCell[ell-1][b]
                        dCell[ell-1][b] = (J@((Weight_before).T@(Dx@delta).reshape(-1, 1))).reshape(-1)
    
    return Wcell, ThetaCell, SCellsmooth, zbarCellsmooth, aCell

def get_inference_error_with_batch_norm(
        Wcell, ThetaCell, SCellsmooth, zbarCellsmooth, aCell, features_test, labels_hat,
        L, softmax, number_of_classes, activation):

    # Testing
    Q = number_of_classes
    yCell = [None]*L 
    zCell = [None]*L 

    N_test = features_test.shape[0]
    test = features_test
    labels_hat = np.zeros(N_test) # used to save the predicted labels for comparison with labels_test
    output = np.zeros((Q, N_test))

    labels_test = labels_test.reshape(-1)
    error = 0
    for n in tqdm(range(N_test)):
        h = test[n] # a column vector
        m = labels_test[n] # labels assumed to be nonnegative: 0,1,2,...
        gamma = np.zeros(Q) # transform the label into a vector with all zeros except at a single position with onme
        gamma[m] = 1 # a vector with unit entry

        yCell[0] = h
        for ell in range(L-1): # forward propagation
            Weight = Wcell[ell]
            theta = ThetaCell[ell]
            y = yCell[ell]
            z = Weight@y 
            S = SCellsmooth[ell+1]
            zbar = zbarCellsmooth[ell+1]
            avec = aCell[ell]
            A = np.diag(avec)
            zx = np.linalg.inv(S)@(z-zbar)
            v = A@zx - theta
            K = v.shape[0]
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
        
        vL = v.copy()
        yL = yCell[-1]

        K = vL.shape[0]
        gamma_hat = np.zeros(K) # K and Q are equal to each other

        if softmax == 0:
            gamma_hat = yL # if no softmax is used
        else:
            gamma_hat = np.exp(vL)/np.exp(vL).sum() # when softmax is used
        
        if softmax == 1: # using softmax
            ax = np.max(gamma_hat) # find location of largest probability
            idx = np.argmax(gamma_hat) 
            labels_hat[n] = idx # location defines the label
            if labels_test[n] != labels_hat[n]:
                error += 1
        else: # no softmax is used; compare each output entry to 1/2
            for k in range(K):
                if activation == "sigmoid":
                    if gamma_hat[k] >= 1/2:
                        output[k, n] = 1
                        labels_hat[n] = k
                    else:
                        output[k, n] = 0 # output vector will have ones and zeros at active attributes
                elif activation == "tanh":
                    if gamma_hat[k] >= 0:
                        output[k, n] = 1
                        labels_hat[n] = k
                    else:
                        output[k, n] = 0 # output vector will have ones and zeros at active attributes
            if labels_test[n] != labels_hat[n]:
                error += 1

    return error

def adjust_lambda(lambda_, z):
    # lambda: heatmap before transformation
    # z : pre-activation signal
    # lambda_hat: heatmap after transformation

    K = max(lambda_.shape)
    lambda_hat = np.zeros(K)

    for k in range(K):
        if lambda_[k] > 0 and z[k] > 0:
            lambda_hat[k] = lambda_[k]
        else:
            lambda_hat[k] = 0

    return lambda_hat

def sign_vector(w):
    M = max(w.shape)
    s = np.zeros(M)
    for m in range(M):
        if w[m] >= 0:
            s[m] = 1
        else:
            s[m] = -1
    return s