# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 22:25:23 2019

@author: Nicco
"""
import numpy as np

class matrixdata: #to build a sparse matrix I need to give 3 array of info (row, column and data)
    def __init__(self): #arrays initialization
        self.row=np.array([],dtype=np.int16) 
        self.col=np.array([],dtype=np.int16)
        self.data=np.array([])
    
    def appendx(self,x,y,dat):
        self.row=np.append(self.row,x)
        self.col=np.append(self.col,y)
        self.data=np.append(self.data,dat)
