#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################################################
###  function [hat_x] = project_simplex(x)
################################################################

# This function projects a vector x onto the simplex (x_m >=0 and sum x_m=1)
# It is based on the algorithm from Duchi et al. (2008), ``Efficient projections 
# onto the $\ell_1-$-ball for learning in high dimensions,''  Proc. ICML, 
# pp. 272-279, Helsinki, Finland.

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

def project_simplex(x):
    M = max(x.shape)
    hat_x = np.zeros(M).reshape(-1,1)

    y = np.sort(x)[::-1]  # sort entries of x in descending order
    z = np.cumsum(y)  # entries with cumulative sum

    rho = 1
    a = np.zeros(M).reshape(-1,1)
    for m in range(M):
        a[m] = y[m] + (1 / (m + 1)) * (1 - z[m])
        if a[m] > 0:
            rho = m + 1  # largest index where a(m) is positive
    lambda_val = (1 / rho) * (1 - z[rho - 1])

    for m in range(M):
        hat_x[m] = max(x[m] + lambda_val, 0)
    
    return hat_x

