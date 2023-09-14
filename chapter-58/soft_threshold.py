#!/usr/bin/env python
# coding: utf-8

# function output = soft_threshold(input,alpha)
# 
# Implements the soft thresholding function; the input can be a matrix
# 
# TEXT: A. H. Sayed, INFERENCE AND LEARNING FROM DATA, Cambridge University Press, 2022.
# 
# DISCLAIMER:  This computer code is  provided  "as is"   without  any guarantees.
# Practitioners  should  use it  at their own risk.  While  the  codes in the text are useful for instructional purposes, they are not intended to serve as examples of full-blown or optimized designs.  The author has made no attempt at optimizing the codes, perfecting them, or even checking them for absolute accuracy. In order to keep the codes at a level  that is  easy to follow by students, the author has often chosen to  sacrifice  performance or even programming elegance in  lieu  of simplicity. Students can use the computer codes to run variations of the examples shown in the text. 
# 
# The Jupyter notebook and python codes are developed by Saba Nasiri. 

# In[1]:


def soft_threshold(inputt, alpha):
    
    M = inputt.shape[0]
    N = inputt.shape[1]
    
    output = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            if inputt[m, n] >= alpha:
                output[m, n] = inputt[m, n] - alpha
            elif inputt[m, n] <= -alpha:
                output[m, n] = inputt[m, n] + alpha
            else:
                output[m, n] = 0


# In[ ]:




