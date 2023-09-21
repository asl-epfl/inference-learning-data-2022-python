#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#############################################
###   function z = hadamard_product(x,y)
##############################################

# Hadamard product of two vectors (elementwise product)

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

def hadamard_product(x, y):
    L = max(x.shape)
    z = np.zeros(L).reshape(-1,1)
    for n in range(L):
        z[n] = x[n] * y[n]
    return z

