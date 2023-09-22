#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######################################################################
###   function [Combination_Matrix, p_Vector]  = 
###   generate_combination_policy(Adjacency,b,Type)
######################################################################

# March 28, 2014 (AHS)

# This function generates various types of combination policies
# and returns the combination matrix and its Perron eigenvector.
#
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

#INPUT
# Adjacency: A[a,b] =  1 if a and b are connected; otherwise zero; it is a square matrix
#
# Type: designates the type of the combination policy desired:
#       `uniform':    uniform or averaging rule
#       `metropolis': Metropolis rule
#       'reldeg':     relative-degree rule
#       'reldegvar':  relative-degree variance rule (only for MSE networks; in
#                     this case, the entries of the vector b should be the inverses of the
#                     sigma_{v,k}^2 at the agents)
#       'relvar':     relative variance rule (only for MSE networks; in
#                     this case, the entries of the vector b should be the inverses of
#                     \mu_k^2 \sigma_{v,k}^2 Tr(R_{u,k}).
#       'hastings':   Hastings rule (in this case, the entries of the vector b should be 
#                     the theta_k^2 = Tr(H^{-1}G_k in the general case; for MSE networks 
#                     with uniform R_{u}, theta_k^2 = 2 \sigma_{v,k}^2 M).
#
# b: auxiliary column vector with the same number of entries as the number of nodes.
#    Its entries are neeeded to construct the `reldegvar', `relvar', and Hastings rules.
#
#OUTPUT
# Combination_Matrix
#
# p_vector: Perron eigenvector of A (a column vector); its entries are positive and add up to one since the topolgy is strongly-connected.
#

import numpy as np

def generate_combination_policy(Adjacency, b, Type):

    A = Adjacency  # Adjacency matrix
    N = max(A.shape)

    # Determine the number of neighbors of each node from the adjacency matrix
    num_nb = np.sum(A, axis=1)

    W = np.zeros((N, N))

    if Type.lower() == 'uniform':
        # Uniform or averaging rule (left-stochastic)
        W = A;
        for k in range(N) :
            W[k,:] = W[k,:]/np.sum(W[k,:])

    elif Type.lower() == 'metropolis':
        # Metropolis rule (doubly-stochastic)
        for k in range(N) :
            for l in range(N):
                W[k, l] = A[k, l] / max(num_nb[k], num_nb[l])
            W[k, k] = 1 + W[k, k] - np.sum(W[k, :])

    elif Type.lower() == 'reldeg':
        # Relative-degree rule (left-stochastic)
        for k in range(N) :
            W[k,:] = (A[k,:]*num_nb.T)/(A[k,:] @ num_nb)
        end

    elif Type.lower() == 'reldegvar':
        # Relative-degree variance rule (left-stochastic)
        wb = num_nb * b
        for k in range(N) :
            W[k,:] = (A[k,:]*wb.T)/(A[k,:] @ wb)

    elif Type.lower() == 'relvar':
        # Relative variance rule (left-stochastic)
        for k in range(N) :
            W[k,:] = (A[k,:]*b.T)/(A[k,:] @ b)

    elif Type.lower() == 'hastings':
        # General Hastings rule (left-stochastic)
        for k in range(N) :
            for l in range(N):
                W[k, l] = A[k, l] * b[k] / max(num_nb[k] * b[k], num_nb[l] * b[l])
            W[k, k] = 1 + W[k, k] - np.sum(W[k, :])

    else:
        print('Unknown rule of combination!')
        return None, None

    Combination_Matrix = W.T  # The desired combination matrix is the transpose of W

    # Finding the Perron eigenvector
    _, D, V = np.linalg.svd(Combination_Matrix)  # Singular value decomposition
    idx = np.argmax(np.abs(D))
    p = V[idx, :]
    p = p / np.sum(p)  # Normalize the sum of its entries to one

    p_Vector = p  # Perron eigenvector

    return Combination_Matrix, p_Vector

