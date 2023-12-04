#!/usr/bin/env python
# coding: utf-8

# function [ws Ps] = lasso_minimizer(alpha, gamma_vec, H)
# 
# Jan. 2019
# 
# This function returns the minimizer of the empirical LASSO risk (l1-regularized least-squares)
# P(w) = alpha\|w\|_1 + (1/N)\sum_{n=0}^{N-1} (gamma(m)-h_m' w)^2
# 
# This code considers an l1-regularized least-squares empirical risk
# and returns its approximate minimizer. This is determined by running a
# full proximal gradient recursion with a small step-size and for a large 
# number of iterations, and then using the limit value as the 
# approximate minimizer.
# 
# TEXT: A. H. Sayed, INFERENCE AND LEARNING FROM DATA, Cambridge University Press, 2022.
# 
# DISCLAIMER:  This computer code is  provided  "as is"   without  any  guarantees.
# Practitioners  should  use it  at their own risk.  While  the  codes in  the text 
# are useful for instructional purposes, they are not intended to serve as examples 
# of full-blown or optimized designs.  The author has made no attempt at optimizing 
# the codes, perfecting them, or even checking them for absolute accuracy. In order 
# to keep the codes at a level  that is  easy to follow by students, the author has 
# often chosen to  sacrifice  performance or even programming elegance in  lieu  of simplicity. Students can use the computer codes to run variations of the examples 
# shown in the text. 
# 
# The Jupyter notebook and python codes are developed by Saba Nasiri. 
# 

# In[52]:


import numpy as np
from soft_threshold import soft_threshold

def lasso_minimizer(alpha, gamma_vec, H):
    # alpha: l1-regularization parameter
    # gamma_vec: Nx1 label vector
    # H: MxN feature matrix; each column is a feature vector

    M = H.shape[0]
    N = H.shape[1]
    mu = 0.1
    
    w = np.zeros((M,1))
    
    for i in np.arange(0, 5000):
        
        s = np.zeros((M,1))
        for n in range(N):
            h_n = H[:, n].reshape(-1, 1)
            gamma_n = gamma_vec[n]
            a = gamma_n - np.matmul(h_n.T, w)
            s = s + 2* a * h_n
            #print(gamma_n)
        
        grad = s/N
        z = w + mu*grad
        w = soft_threshold(z.reshape(-1, 1), mu*alpha)
        
        
    ws = w
    Ps = 0 
    for m in range(N):
        h_m = H[:, m].reshape(-1, 1)
        gamma_m = gamma_vec[m]
        a = gamma_m - np.matmul(h_m.T, ws)
        Ps = Ps + a**2
    
    print(ws.shape)
    Ps = Ps/N
    Ps = Ps + alpha * np.linalg.norm(ws, 1)
    
    
    return ws.reshape(M, ), Ps[0][0]
        


# In[ ]:




