#!/usr/bin/env python
# coding: utf-8

# function Z = hadamard_matrix_division(X,Y)
# 
# This function computes the Hadamard division of two matrices (elementwise division)
# 
# TEXT: A. H. Sayed, INFERENCE AND LEARNING FROM DATA, Cambridge University Press, 2022.
# 
# DISCLAIMER:  This computer code is  provided  "as is"   without  any  guarantees. Practitioners  should  use it  at their own risk.  While  the  codes in  the text are useful for instructional purposes, they are not intended to serve as examples of full-blown or optimized designs.  The author has made no attempt at optimizing the codes, perfecting them, or even checking them for absolute accuracy. In order to keep the codes at a level  that is  easy to follow by students, the author has often chosen to  sacrifice  performance or even programming elegance in  lieu  of simplicity. Students can use the computer codes to run variations of the examples shown in the text. 
# 
# The Jupyter notebook and python codes are developed by Saba Nasiri. 

# In[1]:


import numpy as np

def hadamard_matrix_division(X, Y):
    M = X.shape[0]
    N = X.shape[1]
    
    Z = np.zeros((M, N))
    epsilon = 1 * 10**(-8)
    for m in range(M):
        for n in range(N):
            if Y[m, n] != 0:
                Z[m, n] = X[m, n] / Y[m, n]
            else:
                if X[m, n] == 0:
                    Z[m, n] = 0
                else:
                    Z[m, n] = X[m, n] / epsilon
            
    return Z


# In[ ]:




