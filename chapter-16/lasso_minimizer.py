#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##########################################################################
###    function [ws Ps] = lasso_minimizer(alpha, gamma_vec, H)
##########################################################################

# Jan. 2019

# This function returns the minimizer of the empirical LASSO risk (l1-regularized least-squares)
# P(w) = alpha\|w\|_1 + (1/N)\sum_{n=0}^{N-1} (gamma(m)-h_m' w)^2

# The code considers an l1-regularized least-squares empirical risk
# and returns its approximate minimizer. This is determined by running a
# full proximal gradient recursion with a small step-size and for a large 
# number of iterations, and then using the limit value as the 
# approximate minimizer.

# alpha: l1-regularization parameter
# gamma_vec: Nx1 label vector
# H: MxN feature matrix; each column is a feature vector

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
from soft_threshold import soft_threshold

def lasso_minimizer(alpha, gamma_vec, H):
    M, N = H.shape
    mu = 0.1  # step-size
    w = np.zeros(M).reshape(-1,1)

    for _ in range(5000):
        s = np.zeros(M).reshape(-1,1)
        for n in range(N):  # computing the full gradient
            h_n = H[:, n].reshape(-1,1)  # feature vector
            gamma_n = gamma_vec[n]  # label
            a = gamma_n - np.dot(h_n.T, w)
            s = s + 2 * h_n * a  # subgradient vector for w
        grad = s / N

        z = w + mu * grad
        w = soft_threshold(z, mu * alpha)

    ws = w.copy()

    # Calculate minimum risk value
    Ps = 0
    for m in range(N):
        h_m = H[:, m]  # feature vector
        gamma_m = gamma_vec[m]  # label
        a = gamma_m - np.dot(h_m.T, ws)
        Ps = Ps + a ** 2
    Ps = Ps / N
    Ps = Ps + alpha * np.linalg.norm(ws, 1)

    return ws, Ps