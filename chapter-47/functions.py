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
        ax = int(np.argwhere(x >= cumulative)[-1] + 1) # returns one index of action; the last one
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