#!/usr/bin/env python
# coding: utf-8

# In[ ]:


####################################################################
### function [alpha_value] = find_alpha(H,gamma_vec,w,rho,P,g,q)
####################################################################

# This function finds the parameter alpha via line search to satisfy the weak Wolfe conditions
# H: feature matrix (rows are feature vectors)
# gamma_vec: vector of labels
# w: iterate
# rho: ell_2-regularization factor
# P: risk value at w
# g: gradient at current w
# q: conjugate direction

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
from risk_value import risk_value
from gradP import gradP

def find_alpha(H, gamma_vec, w, rho, P, g, q):
    max_iterations = 20
    lambda_val = 0.0001
    eta = 0.1
    kappa = 0.2
    
    N = max(H.shape)
    M = max(w.shape)
    
    alpha = 1  # start with a large initial value alpha and reduce it successively
    true = 0   # flag; search stops when true = 1
    i = 1
    
    a = np.dot(g.T, q)
    while (true != 1) and (i < max_iterations):
        wnew = w + alpha * q
        Px = risk_value(H, gamma_vec, wnew, rho)  # risk value at updated weight
        gx = gradP(H, gamma_vec, wnew, rho)  # gradient at updated weight
        b = np.dot(gx.T, q)
        
        if (Px <= P + lambda_val * alpha * a) and (b >= eta * a):
            true = 1
        else:
            i = i + 1
            alpha = kappa * alpha
    
    alpha_value = alpha
    return alpha_value

