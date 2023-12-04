#!/usr/bin/env python
# coding: utf-8

# # function s = sign_vector(w)

# ### This function returns a sign column vector with +- 1 entries
# 
# TEXT: A. H. Sayed, INFERENCE AND LEARNING FROM DATA, Cambridge University Press, 2022.

# <div style="text-align: justify">
# DISCLAIMER:  This computer code is  provided  "as is"   without  any  guarantees.
# Practitioners  should  use it  at their own risk.  While  the  codes in  the text 
# are useful for instructional purposes, they are not intended to serve as examples 
# of full-blown or optimized designs.  The author has made no attempt at optimizing 
# the codes, perfecting them, or even checking them for absolute accuracy. In order 
# to keep the codes at a level  that is  easy to follow by students, the author has 
# often chosen to  sacrifice  performance or even programming elegance in  lieu  of 
# simplicity. Students can use the computer codes to run variations of the examples 
# shown in the text. 
# </div>
# 
# The Jupyter notebook and python codes are developed by Saba Nasiri. 

# required libraries:
#     
# 1. numpy

# In[8]:


import numpy as np


# In[9]:


def sign_vector(w):
    M = max(w.shape)
    s = np.zeros((M,1))
    for m in range(M):
        if w[m]>=0 :
            s[m] = 1
            
        else:
            s[m] = -1
            
            
    return s


# In[ ]:




