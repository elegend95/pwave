# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:05:58 2019

@author: Nicco
"""
from __future__ import division
import numpy as np
import functions as fx
from numba import njit
'''
discrete fourier transform of vectors
'''

fourt=fx.discretefourier(Ltilde,L)
vecsk=np.matmul(fourt,vecs[Ltilde**2:,Ltilde**2:])

mat=fx.angumomk(Ltilde)

angmo=0
for i in range(Ltilde**2):
    angmo+=np.abs(fx.expvalue(mat,vecsk[:,i]))

print(angmo)


@njit
def discretefouriercaz(vecs,Ltilde):
    vecsk=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex64)
    for i in range(Ltilde**2):
        for k in range(Ltilde**2):
            ky,kx=(2*np.pi/L)*(fx.indtocord(k,Ltilde)-(Ltilde-1)/2)
            for j in range(Ltilde**2):
                y,x=fx.indtocord(j,Ltilde)
                vecsk[k,i]+=np.exp(-1j*(kx*x+ky*y))*vecs[j+Ltilde**2,i+Ltilde**2]
    return vecsk

def 