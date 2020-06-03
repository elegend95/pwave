# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:02:11 2019

@author: Nicco
"""
from scipy import sparse as sprs
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

def sparseham1d(sizeham,t,mu):
    matr=matrixdata()
    for i in range(sizeham):
        matr.appendx(i,i,-mu) #set chem pot on diagonal
        matr.appendx(i,(i+1)%sizeham,-t*np.exp(10.**-4*1.j)) #set hopping left
        matr.appendx(i,(i-1)%sizeham,-t*np.exp(10.**-4*-1.j)) #set hopping right
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def sparsemom1d(sizeham):
    matr=matrixdata()
    for i in range(sizeham):
        matr.appendx(i,i,0)
        matr.appendx(i,(i+1)%sizeham,1j)
        matr.appendx(i,(i-1)%sizeham,-1j)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def density1d(i,vecs,Ltilde):
    return sum(np.conj(vecs[i,Ltilde**2:])*vecs[i,Ltilde**2:])