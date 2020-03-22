# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:02:11 2019

@author: Nicco
"""
from scipy import sparse as sprs
import numpy as np
import classes as cl

def sparseham1d(sizeham,t,mu):
    matr=cl.matrixdata()
    for i in range(sizeham):
        matr.appendx(i,i,-mu) #set chem pot on diagonal
        matr.appendx(i,(i+1)%sizeham,-t*np.exp(10.**-4*1.j)) #set hopping left
        matr.appendx(i,(i-1)%sizeham,-t*np.exp(10.**-4*-1.j)) #set hopping right
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def sparsemom1d(sizeham):
    matr=cl.matrixdata()
    for i in range(sizeham):
        matr.appendx(i,i,0)
        matr.appendx(i,(i+1)%sizeham,1j)
        matr.appendx(i,(i-1)%sizeham,-1j)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def density1d(i,vecs,Ltilde):
    return sum(np.conj(vecs[i,Ltilde**2:])*vecs[i,Ltilde**2:])