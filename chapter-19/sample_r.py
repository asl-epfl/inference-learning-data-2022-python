#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#############################################################
###  function index = sample_r(r)
#############################################################

# r: NX1 vector of probabilities adding up to one
# output: a sample selected from the input vector according to distribution r

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

# The python codes are developed by Semanur Avşar 

import numpy as np

def sample_r(r):
    N = max(r.shape)
    x = np.random.rand()  # a random variable between 0 and 1
    
    if x <= r[0]:
        index = 0
    else:
        _sum = r[0]
        for n in range(N-1):
            if ( (x >= _sum) and (x < (_sum + r[n+1])) ) :
                index = n + 1
            _sum += r[n+1]
    
    return index

