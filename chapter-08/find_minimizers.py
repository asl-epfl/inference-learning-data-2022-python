#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################################################
###  function [ws_logistic ws_hinge Ps_logistic Ps_hinge] = 
###  find_minimizers(rho, gamma,h_logistic, h_hinge)
##################################################################

# Jan. 2019

# This function returns approximate minimizers for the logistic and 
# hinge functions used in examples throughout the text.

# This code uses a gradient/subgradient descent implementation with small
# step-sizes and a large number of iterations.

# The function returns minimizers and minimum values

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

import numpy as np

def find_minimizers(rho, gamma, h_logistic, h_hinge):
    
    # g(z) = (rho/2) \|w\|^2 + ln(1+exp(-gamma h'w);  logistic
    # g(z) = (rho/2) \|w\|^2 + max(0,1-gamma h' w);  hinge
                               
    M = len(h_logistic)
    mu_logistic = 0.1  # step-sizes
    mu_hinge = 0.001

    w_logistic = np.zeros(M)
    w_hinge = np.zeros(M)

    for i in range(50000):
        a = np.exp(-gamma * np.dot(h_logistic.T, w_logistic))
        s = gamma * h_logistic * a / (1 + a)  # gradient vector for w

        ah = (gamma * np.dot(h_hinge.T, w_hinge) <= 1)
        sh = gamma * h_hinge * ah

        w_logistic = w_logistic - mu_logistic * rho * w_logistic + mu_logistic * s
        w_hinge = w_hinge - mu_hinge * rho * w_hinge + mu_hinge * sh

    ws_logistic = w_logistic
    ws_hinge = w_hinge

    # Calculate minimum risk value, denoted by Ps.
    a = 1 + np.exp(-gamma * np.dot(h_logistic.T, ws_logistic))
    Ps_logistic = (rho / 2) * (np.linalg.norm(ws_logistic) ** 2) + np.log(a)

    ah = np.maximum(0, 1 - gamma * np.dot(h_hinge.T, ws_hinge))
    Ps_hinge = (rho / 2) * (np.linalg.norm(ws_hinge) ** 2) + ah

    # testing the optimality of ws-logistic
    a = np.exp(-gamma * np.dot(h_logistic.T, ws_logistic))
    s = rho * ws_logistic - gamma * h_logistic * a / (1 + a)  # gradient vector

    print('this value is printed from within the function find_minimizers')
    print('its value should be close to zero to indicate that ws_logistic is minimizer')
    print(s)  # ws-logistic is optimal if this variable is ~ 0

    return ws_logistic, ws_hinge, Ps_logistic, Ps_hinge


# In[ ]:


################################################################
###  function [ws_logistic ws_hinge Ps_logistic Ps_hinge] = 
###  find_minimizers(rho, gamma,h_logistic, h_hinge)
##################################################################

# Jan. 2019

# This function returns approximate minimizers for the logistic and 
# hinge functions used in examples throughout the text.

# This code uses a gradient/subgradient descent implementation with small
# step-sizes and a large number of iterations.

# The function returns minimizers and minimum values

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

