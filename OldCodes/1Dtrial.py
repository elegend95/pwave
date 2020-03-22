# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:43:36 2019

@author: Nicco
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as alg
import functions1d as fx
from scipy import sparse as sprs

#trial of Code1 functions for 1D systems

#parameters inizialization
L=10 #matrix edge
Ltilde=10 #need a matrix element for every lattice point
sizeham=Ltilde
a=L/Ltilde
m=1
t=1./(2*m)
mu=-2*t
delta=3
#lattice step

#write explicitly lattice spacing dependence
#parameters

#hamiltonian definition
ham1d=a**-1*fx.sparseham1d(sizeham,t,mu)
px=fx.sparsemom1d(sizeham)
    
vals,vecs=sprs.linalg.eigsh(a*ham1d,k=sizeham-2,which='SM')
#valsl,vecsl=sprs.linalg.eigsh(ham1d,k=sizeham/2,which='SM')
#vals=np.append(valsl,valsu)
#vecs=np.append(vecsl,vecsu)
#vals,vecs=alg.eigh(ham1d.toarray())
##impulse operator definition 1D
#px=np.zeros((sizeham,sizeham),dtype=np.csingle)
#for i in range(sizeham):
#    px[i][i]=0
#    px[i][(i+1)%sizeham]=1.j
#    px[i][(i-1)%sizeham]=-1.j
    
k=np.zeros(len(vals))
#for j in range(L-2):
#    k[j]=fx.expvalue(px.toarray(),vecs[:,j])

#ka=np.sqrt(np.arcsin(k/2)**2)
for j in range(len(vals)):
    k[j]=-1j*np.log(vecs[1,j]/vecs[0,j])/L

plt.figure(1)
plt.plot(k,vals,'bo')
asc=np.linspace(-np.pi/L,np.pi/L,100)
plt.plot(asc,-2*t*(np.cos(asc*L)-1))

#
#for a in [8,4,2,1,0.5]:
#    asc=np.linspace(-np.pi/a,np.pi/a,100)
#    plt.plot(asc,-(np.cos(asc*a)-1)/a**2)
#     
#plt.plot(asc,(asc**2)/2)