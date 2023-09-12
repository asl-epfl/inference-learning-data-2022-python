import numpy as np

def select_action(policy):
    # policy = pi(a|s)

    number_of_actions = max(policy.shape)
    pi_vec = []
    a_vec = []
    counter = 0
    for a in range(number_of_actions): # let us isolate the actions with nonzero probabilities
        if policy[a] != 0:
            pi_vec.append(policy[a])
            a_vec.append(a)
            counter += 1

    pi_vec = np.array(pi_vec)
    a_vec = np.array(a_vec)
    cumulative = np.cumsum(pi_vec) # vector of cumulative probabilities
    x = np.random.rand() # a random number in [0, 1]
    if x < cumulative.min():
        act = a_vec[0]
    else:
        ax = int(np.argwhere(x >= cumulative)[-1]) + 1 # returns one index of action; the last one
        act = a_vec[ax]
    return int(act)

def select_next_state(kernel):
    # kernel = P(s, a, :)

    number_of_states = max(kernel.shape)
    counter = 0
    p_vec = []
    s_vec = []
    for sprime in range(number_of_states): # let us isolate the states with nonzero transition probabilities
        if kernel[sprime] != 0:
            p_vec.append(kernel[sprime])
            s_vec.append(sprime)
            counter += 1
    
    p_vec = np.array(p_vec)
    s_vec = np.array(s_vec)
    cumulative = np.cumsum(p_vec) # vector os cumulative probabilities
    x = np.random.rand() # a random number in [0, 1]
    if x < cumulative.min():
        sprime = s_vec[0]
    else:
        sx = int(np.argwhere(x>=cumulative)[-1] + 1) # returns one index of action; the last one
        sprime = s_vec[sx]
    return int(sprime)

def feed_back(Wcell,ThetaCell,yCell,zCell,L,type_activation,sigmaL,mu,rho,use_theta):
    # feedback iteration of backpropagation

    Wcellf = [np.zeros(1)]*(L-1)
    ThetaCellf = [np.zeros(1)]*(L-1)
    dCellf = [np.zeros(1)]*L

    dCellf[-1] = sigmaL

    for ell in range(L-1, 0, -1):
        Weight_before = Wcell[ell-1]
        if use_theta:
            theta_before = ThetaCell[ell-1]
        else:
            theta_before = np.zeros(len(Weight_before))
        y = yCell[ell-1]
        sigma = dCellf[ell]
        Weight = (1-2*mu*rho)*Weight_before - mu*sigma.reshape(-1, 1)@y.reshape(1, -1)
        if use_theta == 1:
            theta = theta_before + mu*sigma
        else:
            theta = np.zeros(len(Weight_before))
        
        Wcellf[ell-1] = Weight
        ThetaCell[ell-1] = theta
        if ell >= 2: # computing next delta only for ell >=2
            z = zCell[ell-1]
            K = len(z)
            J = np.zeros((K, K))
            type_ = type_activation[ell-1]

            if type_ == 0: #linear
                J = np.eye(K)
            elif type_ == 1: #sigmoid
                for k in range(K):
                    f = 1/(1+np.exp(-z[k]))
                    J[k, k] = f*(1-f)
            elif type_ == 2: #tanh
                for k in range(K):
                    b = np.exp(z[k]) + np.exp(-z[k])
                    J[k, k] = 4/(b**2)
            elif type_ == 3: #rectifier
                for k in range(K):
                    if z[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                        J[k,k] = 0
                    elif z[k] >0:
                        J[k, k] = 1
                    elif z[k] < 0:
                        J[k, k] = 0
            dCellf[ell-1] = J@(Weight_before.T@sigma)

    return Wcellf, ThetaCellf

def feed_forward(Wcell,ThetaCell,L,type_activation,h,use_theta):

    yCellf = [np.zeros(1)]*L
    zCellf = [np.zeros(1)]*L

    y = h
    yCellf[0] = y

    for ell in range(L-1):
        Weight = Wcell[ell]
        if use_theta:
            theta = ThetaCell[ell]
        else:
            theta = np.zeros(len(Weight))
        y = yCellf[ell]
        z = Weight@y - theta
        zCellf[ell+1] = z
        K = len(z)
        y = np.zeros(K)
        type_ = type_activation[ell+1]

        if type_ == 0: # linear
            y = z
        elif type_ == 1: # sigmoid
            for k in range(K):
                y[k] = 1/(1+np.exp(-z[k]))
        elif type_ == 2: # tanh
            for k in range(K):
                a = np.exp(z[k]) - np.exp(-z[k])
                b = np.exp(z[k]) + np.exp(-z[k])
                y[k] = a/b
        elif type_ == 3: # rectifier
            for k in range(K):
                y[k] = max([0, z[k]])
        yCellf[ell+1] = y

    return yCellf, zCellf

def compute_policy(F,theta,T,NA,NS,Pi):

    # F: matrix of feature vectors f_{s,a}
    # theta: parameter for pi(a|h;theta)
    # T: size of theta
    # NA: number of actions
    # NS: number of states
    # Pi: original pi(a|s) used to determine which actions are permissible

    sum_ = np.zeros(NS)
    Pi_theta = np.zeros((NA, NS))
    for s in range(NS):
        pi_vec = Pi[:, s] # helps reveal which actions are permissible at state s
        for a in range(NA): # iterate over actions
            if pi_vec[a] > 0:
                row_idx = (s-1)*NA + a # index of row in F corresponding to (s, a)
                f = F[int(row_idx), :].T # feature vector
                sum_[s] += np.exp(f.T@theta)
    
    for s in range(NS):
        pi_vec = Pi[:, s]
        for a in range(NA):
            if pi_vec[a] > 0:
                row_idx = (s-1)*NA + a #index of row in F corresponding to (s, a)
                f = F[int(row_idx), :].T
                Pi_theta[a, s] = np.exp(f.T@theta)/sum_[s]
    
    return Pi_theta

def compute_gradient_log_pi(F,Pi_theta,s,a,NA):
    # F: matrix of feature vectors f_{s,a}
    # Pi_theta: Gibbs distribution matrix NA x NS
    # s: state
    # action
    # NA: number of actions

    pi_vec = Pi_theta[:, s] # pi(a|s; theta)
    row_idx = (s-1)*NA + a # index of row in F corresponding to (s, a)
    f = F[int(row_idx), :].T # feature vector is at row of index sxa; f_{s,a}
    fs_bar = 0
    for aprime in range(NA):
        row_idx_prime = (s-1)*NA + aprime
        faprime = F[int(row_idx_prime), :].T # f_{s, a'}
        fs_bar += pi_vec[aprime]*faprime
    g_vec = f - fs_bar
    return g_vec

def softmax_layer(z):
    
    Qc = max(z.shape)
    output = np.zeros(Qc)
    zc = z - max(z) # to avoid overflow/underflow
    
    sum_ = 0
    for q in range(Qc):
        sum_ += np.exp(zc[q])
    
    for q in range(Qc):
        output[q] = np.exp(zc[q])/sum_
    
    return output

def softmax_permissible(z, flag):
    
    Qc = max(z.shape)
    Qr = sum(flag) # number of unit entries
    z_reduced = np.zeros(Qr)
    flag_reduced = np.zeros(Qr)
    
    counter = 0
    for f in range(Qc):
        if flag[f] > 0:
            z_reduced[counter] = z[f]
            flag_reduced[counter] = f
            counter += 1
    
    zc = z_reduced - max(z_reduced) # to avoid overflow/underflow
    oc = np.zeros(Qr)
    
    sumx = 0
    for q in range(Qr):
        sumx += np.exp(zc[q])
    
    for q in range(Qr):
        oc[q] = np.exp(zc[q])/sumx 
    
    output = np.zeros(Qc)
    for q in range(Qr):
        fx = int(flag_reduced[q])
        output[fx] = oc[q]
        
    return output

def feed_forward_permissible(Wcell,ThetaCell,L,type_activation,h,use_theta, flag):

    yCellf = [np.zeros(1)]*L
    zCellf = [np.zeros(1)]*L

    y = h
    yCellf[0] = y

    for ell in range(L-1):
        Weight = Wcell[ell]
        if use_theta:
            theta = ThetaCell[ell]
        else:
            theta = np.zeros(len(Weight))
        y = yCellf[ell]
        z = Weight@y - theta
        zCellf[ell+1] = z
        K = len(z)
        y = np.zeros(K)
        type_ = type_activation[ell+1]

        if type_ == 0: # linear
            y = z
        elif type_ == 1: # sigmoid
            for k in range(K):
                y[k] = 1/(1+np.exp(-z[k]))
        elif type_ == 2: # tanh
            for k in range(K):
                a = np.exp(z[k]) - np.exp(-z[k])
                b = np.exp(z[k]) + np.exp(-z[k])
                y[k] = a/b
        elif type_ == 3: # rectifier
            for k in range(K):
                y[k] = max([0, z[k]])
        elif type_ == 4:
            y = softmax_permissible(z, flag)
        yCellf[ell+1] = y

    return yCellf, zCellf

def feed_back_batch(Wcell,ThetaCell,saved_y_values,saved_z_values,saved_sigma,L,type_activation,mu,rho,use_theta,N):

    Wcellf = [np.zeros(1)]*(L-1)
    ThetaCellf = [np.zeros(1)]*(L-1)

    dCellf = saved_sigma # boundary sigmaL value for all n; dCellf[n, L]

    for ell in range(L-1, 0, -1):
        Weight_before = Wcell[ell-1]
        if use_theta == 1:
            theta_before = ThetaCell[ell-1]
        else:
            theta_before = np.zeros(len(Weight_before))
        
        a, b = Weight_before.shape
        g1 = np.zeros((a, b))
        g2 = np.zeros(a)

        for n in range(N):
            yCell = saved_y_values[n] # a cell
            yell = yCell[ell-1]
            sigma = dCellf[n][ell-1]

            g1 += sigma*yell.T
            g2 += sigma

        g1 = g1/N
        g2 = g2/N

        Weight = (1-2*mu*rho)*Weight_before + mu*g1 
        if use_theta == 1:
            theta = theta_before - mu*g2
        else:
            theta = np.zeros(len(Weight_before))
        
        Wcellf[ell-1] = Weight
        ThetaCell[ell-1] = theta

        if ell >= 2: # computing next delta only for ell >=2
            for n in range(N):
                zCell = saved_z_values[n]
                z = zCell[ell-1]
                K = len(z)
                J = np.zeros((K, K))

                sigma = dCellf[n][ell]
                type_ = type_activation[ell-1]

                if type_ == 0: #linear
                    J = np.eye(K)
                elif type_ == 1: #sigmoid
                    for k in range(K):
                        f = 1/(1+np.exp(-z[k]))
                        J[k, k] = f*(1-f)
                elif type_ == 2: #tanh
                    for k in range(K):
                        b = np.exp(z[k]) + np.exp(-z[k])
                        J[k, k] = 4/(b**2)
                elif type_ == 3: #rectifier
                    for k in range(K):
                        if z[k] == 0: # set, by convention, f'(z) to zero at z=0 for the rectifier function
                            J[k,k] = 0
                        elif z[k] >0:
                            J[k, k] = 1
                        elif z[k] < 0:
                            J[k, k] = 0
                dCellf[n][ell-1] = J@(Weight_before.T@sigma)
    
    return Wcellf, ThetaCellf