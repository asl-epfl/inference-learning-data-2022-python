#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##########################################################################
### function [ws Ps thetas] = svm_minimizer(rho, gamma_vec, H,flag)
##########################################################################

# Jan. 2019

# This function returns the minimizer of the empirical SVM risk:
# P(w) = rho\|w\|^2 + (1/N) sum_{n=0}^{N-1} max(0, 1- gamma(m) h_m' w)

# The code considers an l2-regularized empirical SVM risk
# and returns its minimizer. This is determined by running a
# full subgradient recursion with a small step-size and for a large 
# number of iterations, and then using the limit value as the 
# approximate minimizer.

# rho: l2-regularization parameter
# gamma_vec: Nx1 label vector
# H: MxN feature matrix; each column is a feature vector

# flag = 1 ==> estimate both ws and thetas
# flag = 0 ==> ignore offset and generate only ws

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

# python codes are developed by Semanur Avşar

import numpy as np

def svm_minimizer(rho, gamma_vec, H, flag):
    M, N = H.shape
    mu = 0.001  # step-size

    w = np.zeros(M).reshape(-1,1)
    theta = 0

    for i in range(50000):
        s = np.zeros(M).reshape(-1,1)
        s2 = 0
        for n in range(N):
            h_n = H[:, n].reshape(-1,1)  # feature vector
            gamma_n = gamma_vec[n]  # label
            a = (gamma_n * (np.dot(h_n.T, w) - theta) <= 1)
            s += gamma_n * h_n * a  # subgradient vector for w
            s2 += flag * gamma_n * a  # subgradient for theta
        grad = s / N
        grad2 = s2 / N

        w = w - mu * (2 * rho) * w + mu * grad
        theta = theta - flag * mu * gamma_n * grad2  # flag=0 ==> theta stays at 0

    ws = w.copy()
    thetas = theta

    # Calculate minimum risk value Ps
    Ps = 0
    for m in range(N):
        h_m = H[:, m].reshape(-1,1)  # feature vector
        gamma_m = gamma_vec[m]  # label
        a = max(0, 1 - gamma_m * np.dot(h_m.T, ws))
        Ps += a
    Ps = Ps / N
    Ps = Ps + rho * np.linalg.norm(ws, 2)**2

    return ws, Ps, thetas

