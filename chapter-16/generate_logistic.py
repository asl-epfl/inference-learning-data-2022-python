#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##########################################################################
###  function [p gamma_vec H] = generate_logistic(N,wo,theta)
##########################################################################

# This function generates N data pairs (gamma,h) that follow a logistic model with
# parameter wo of size Mx1  and offset theta. The feature vectors are Gaussian distributed.

# offset parameter theta should be small, say, p= 0.1 or smaller, to help ensure p close to 0.5

# TEXT: A. H. Sayed, INFERENCE AND LEARNING FROM DATA, 
#       Cambridge University Press, 2022.

# DISCLAIMER:  This computer code is  provided  "as is"   without  any  guarantees.
# Practitioners  should  use it  at their own risk.  While  the  codes in  the text 
# are useful for instructional purposes, they are not intended to serve as examples 
# of full-blown or optimized designs.  The author has made no attempt at optimizing 
# the codes, perfecting them, or even checking them for absolute accuracy. In order 
# to keep the codes at a level  that is  easy to follow by students, the author has 
# often chosen to  sacrifice  performance or even programming elegance in  lieu  of 
# simplicity. Students can use the computer codes to run variations of the examples 
# shown in the text. 

# the python code is developed by Semanur Avşar

import numpy as np

def generate_logistic(N, wo, theta):
    M = max(wo.shape)
    gamma_vec = np.zeros(N).reshape(-1,1)
    H = np.random.randn(M, N)  # each column is a feature vector
    
    counter = 0
    for n in range(N):
        h = H[:, n].reshape(-1,1)  # feature vector
        px = 1 / (1 + np.exp(-(np.dot(h.T, wo) - theta)))  # probability of gamma = +1
        x = np.random.rand()
        if x <= px:
            gamma_vec[n] = +1
            counter += 1
        else:
            gamma_vec[n] = -1
            
    p = counter / N  # fraction of +1's in the result
    return p, gamma_vec, H