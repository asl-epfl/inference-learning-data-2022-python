#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#######################################################################
###   function plot_topology(Adjacency,Coordinates,Color)
#######################################################################

# April 2, 2014 (AHS)

# This function plots a network topology

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
# Adjacency: size NxN; A[a,b] =  1 if a and b are connected; otherwise zero.
#
# Coordinates: Nx2 matrix containing the (x,y) location coordinates of the agents in the square region [0,1.2]x[0,1.2]; 
#              each row corresponds to one agent
#
# Color: a vector of size Nx1. Location k in this vector is set to:
#        0 if the corresponding agent should have one color (yellow)
#        1 if the corresponding agent should have a second color (red)
#        2 if the corresponding agent should have a third color (green)


import numpy as np
import matplotlib.pyplot as plt

def plot_topology(Adjacency, Coordinates, Color):

    A = Adjacency  # Adjacency matrix
    N = max(A.shape)  # Number of agents

    x_coordinates = Coordinates[:, 0]  # x-coordinates of agents
    y_coordinates = Coordinates[:, 1]  # y-coordinates of agents

    plt.figure()
    plt.axis([0, 1.2, 0, 1.2])
    plt.axis('square')
    plt.grid(True)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')

    for k in range(N):
        for l in range(N):
            if A[k, l] > 0:
                plt.plot([x_coordinates[k], x_coordinates[l]], [y_coordinates[k], y_coordinates[l]], 'b-', linewidth=1.5)

    for k in range(N):
        if Color[k] == 0:  # Yellow
            plt.plot(x_coordinates[k], y_coordinates[k], 'o', markeredgecolor='b', markerfacecolor='y', markersize=10)
        elif Color[k] == 1:  # Red
            plt.plot(x_coordinates[k], y_coordinates[k], 'o', markeredgecolor='b', markerfacecolor='r', markersize=10)
        else:  # Green
            plt.plot(x_coordinates[k], y_coordinates[k], 'o', markeredgecolor='b', markerfacecolor='g', markersize=10)

    for k in range(N):
        plt.text(x_coordinates[k] + 0.03, y_coordinates[k] + 0.03, str(k + 1), fontsize=7)

    plt.savefig("figs/fog-7.pdf", format="pdf", bbox_inches="tight")

    plt.show()

