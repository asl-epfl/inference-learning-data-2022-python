#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#############################################################################################
###   function [Adjacency,Laplacian,Algebraic_Connectivity,Degree_Vector,Coordinates] = 
###   generate_topology(Num_nodes,Type,Parameter)
#############################################################################################

# March 27, 2014 (AHS)

# This function generates a random network topology with N agents and 
# returns its adjacency and Laplacian matrices, the algebraic connectivity,
# and a column vector containing the degrees of the various nodes.
# The function also plots the network if it turns out to be connected.
# The function allows the network to be generated according to two
# criteria, as explained below.
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
# Num_nodes: number of nodes in the network

# Type = 1: Nodes that are within a certain radius from each other are declated neighbors.
# Type = 2: Two nodes are declared neighbors probabilisitically according to a binomial distribution.

# Parameter (always positive and smaller than one):
#      for Type=1, this variable refers to the radius: in the square region [0,1.2]x[0,1.2], nodes within 
#          this value from each other are declared to be neighbors, e.g., radius = 0.3.
#      for Type 2, this variable refers to the probability of establishing a connection. Nodes a and b will
#          be neighbors with this probability value.

#OUTPUT
# Adjacency: A[a,b] =  1 if a and b are connected; otherwise zero.

# Laplacian: L[a,b] = -1 if a and b are connected; L[a,a] = degree(a)-1.
# The degree of an agent a is equal to the number of its neighbors including itself.

# Algebraic_Connectivity: smallest second eigenvalue of L; if nonzero, then the network is connected.
#                         Since in the topology construction, every node is assumed connected to itself, 
#                         the resulting topology will be strongly-connected.
#
# Degree_Vector: a vector containing the degrees (number of neighbors) of the nodes
#
# Coordinates: Nx2 matrix containing the (x,y) location coordinates of the agents; each row corresponds to one agent

import numpy as np

def generate_topology(Num_nodes, Type, Parameter):
    N = Num_nodes  # Number of nodes.
    A = np.zeros((N, N))  # Adjacency matrix.
    L = np.zeros((N, N))  # Laplacian matrix.

    if Type == 1:
        r = Parameter  # Nodes within this radius from each other are declared to be neighbors.
    else:
        p = Parameter  # Nodes k and l are declared neighbors according to a binomial distribution with probability p.

    # We first generate N random (x,y) coordinates in the square region [0,1.2]x[0,1.2]
    x_coordinates = np.random.rand(N) + 0.1
    y_coordinates = np.random.rand(N) + 0.1
    Coordinates = np.column_stack((x_coordinates, y_coordinates))

    # We next determine which nodes are neighbors of each other and find the adjacency matrix.
    if Type == 1:  # distance criterion
        for k in range(N):
            for l in range(N):
                d = np.sqrt((x_coordinates[k] - x_coordinates[l]) ** 2 + (y_coordinates[k] - y_coordinates[l]) ** 2)
                if d <= r:
                    A[k, l] = 1  # set entry in adjacency matrix to one if nodes k and l should be neighbors.

    if Type == 2:  # binomial criterion
        for k in range(N):
            A[k, k] = 1  # a node is always connected to itself in this construction
            for l in range(k + 1, N):
                b = np.random.rand()  # generate a uniform random variable in the interval [0,1]
                if b <= p:  # if b falls within the interval [0,p], then we connect the nodes
                    A[k, l] = 1  # set entry in adjacency matrix to one if nodes k and l should be neighbors.
                    A[l, k] = 1

    Adjacency = A  # adjacency matrix.

    # We determine the number of neighbors of each node from the adjacency matrix
    num_nb = np.zeros(N).reshape(-1,1)
    for k in range(N):
        num_nb[k] = np.sum(A[k, :])
    Degree_Vector = num_nb  # vector of degrees for the various nodes

    # We now compute the Laplacian matrix L and check if the network is connected
    # by verifying whether the second smallest eigenvalue of L is positive.
    for k in range(N):
        L[k, k] = max(0, np.sum(A[k, :]) - 1)  # set diagonal entry to zero if degree-1 for node k is negative.
        for l in range(k + 1, N):
            L[k, l] = -1 * A[k, l]
            L[l, k] = -1 * A[l, k]

    sigma = np.linalg.svd(L, compute_uv=False)  # vector of singular values of L.

    Laplacian = L  # Laplacian matrix
    Algebraic_Connectivity = sigma[N - 2]  # algebraic connectivity

    if sigma[N - 2] < 1e-4:  # checking if the second smallest singular value is positive (sufficiently away from zero).
        return None  # network is not connected; returns None to indicate a failure.
    else:
        return Adjacency, Laplacian, Algebraic_Connectivity, Degree_Vector, Coordinates

# else
#     # plot the network
#     figure
#     hold on
#     for k=1:N
#         for l=1:N
#             if A(k,l)>0
#                 plot([x_coordinates(k),x_coordinates(l)],[y_coordinates(k),y_coordinates(l)],'b-','LineWidth',1.5);
#             end
#         end
#     end
#     plot(x_coordinates,y_coordinates,'o','MarkerEdgeColor','b','MarkerFaceColor','y','MarkerSize',10);
#     axis([0,1.2,0,1.2]);
#     axis square
#     grid
#     xlabel('x-coordinate')
#     ylabel('y-coordinate')
# 
#     for k=1:N
#       text(x_coordinates(k)+0.03,y_coordinates(k)+0.03,num2str(k),'Fontsize',7);
#     end  

