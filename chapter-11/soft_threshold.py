#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########################################################
### function output = soft_threshold(input,alpha)
#########################################################

# This function implements the soft thresholding function
# The input can be a matrix.

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

def soft_threshold(input_data, alpha):

    M, N= input_data.shape
    output = np.zeros((M, N))
    
    for m in range(M):
        for n in range(N):
            if input_data[m, n] >= alpha:
                output[m, n] = input_data[m, n] - alpha
            elif input_data[m, n] <= -alpha:
                output[m, n] = input_data[m, n] + alpha
            else:
                output[m, n] = 0
    
    return output

