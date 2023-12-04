#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##########################################
###     function s = sign_vector(w)
##########################################

# sign function; returns a sign column vector with +- 1 entries

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

def sign_vector(w):
    M = max(w.shape)  # w is a column vector
    s = np.zeros(M).reshape(-1,1)

    for m in range(M):
        if w[m] >= 0:
            s[m] = 1
        else:
            s[m] = -1

    return s