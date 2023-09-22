#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########################################################################
###   function [p_vector]  = find_Perron_vector(combination_policy)
#########################################################################

# April 5, 2014 (AHS)

# This function finds the Perron vector of a combination policy, namely,
# the vector that satisfies
# Ap=p, 1^T p =1, p_k>0
#
# If the matrix A is primitive, then all entries p_k are strictly positive.
#
# TEXT: A. H. Sayed, INFERENCE AND LEARNING FROM DATA, 
#       Cambridge University Press, 2022.
#
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
import scipy.linalg as la

def find_Perron_vector(combination_policy):
    
#INPUT
# combination_policy: NxN stochastic matrix (left, doubly, or right).
#
#OUTPUT
# p_vector: Perron eigenvector of A (a column vector); its entries are positive and add up to one since the topolgy is strongly-connected.
#

    A = combination_policy  # Combination matrix
    N = max(A.shape)
    eigenvalues, eigenvectors = la.eig(A)  # Eigenvalue decomposition
    idx = np.argmax(np.abs(eigenvalues))  # Index of maximum magnitude eigenvalue of A
    p = eigenvectors[:, idx]  # Extracting the corresponding eigenvector
    p = p / np.sum(p)  # Normalizing the sum of its entries to one

    p_vector = p  # Perron eigenvector

    r = np.ones(N)
    for i in range(1000):
        r = A.dot(r)

    return p_vector, r / np.linalg.norm(r)

