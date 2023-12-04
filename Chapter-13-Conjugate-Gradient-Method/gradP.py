#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#################################################################
### function [grad_value] = gradP(H,gamma_vec,w,rho)
#################################################################

# This function computes the gradient of the logistic risk
# H: feature matrix (rows are feature vectors)
# gamma_vec: vector of labels
# w: iterate
# rho: ell_2-regularization factor

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

def gradP(H, gamma_vec, w, rho):
    N = max(H.shape)
    M = max(w.shape)
    g = np.zeros(M).reshape(-1,1)
    
    for m in range(N):
        h_m = H[:, m].reshape(-1,1)  # feature vector
        gamma_m = gamma_vec[m]  # label
        a = 1 + np.exp(gamma_m * np.dot(h_m.T, w))
        g += gamma_m * h_m * (1 / a)
    
    grad_value = 2 * rho * w - (1 / N) * g
    return grad_value

