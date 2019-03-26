# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:24:38 2019

@author: Nicco
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functions as fx
from scipy import sparse as sprs
from scipy.sparse import linalg


#inefficient hamiltonian building
#building of hamiltonian with given parameters
for i in range(sizeham): #search a site i on lattice
    ham[i][i]=-mu #puts chemical potential on diagonal
    nearn=fx.nearneigh2D(i,L) #list of i point neighbors
    for j in range(len(nearn)):
        ham[i][nearn[j]]=-t #assigns hopping term on neighbor terms

#impulse operator definition 1D
px=np.zeros((sizeham,sizeham),dtype=np.csingle)
for i in range(sizeham):
    px[i][i]=0
    px[i][(i+1)%sizeham]=1.j
    px[i][(i-1)%sizeham]=-1.j

#construction of momentum operator, gives problem bc we have to do arcsin at the end and there is doubling problem
pxss=fx.sprspx(L) #construction of upper block of x direction momentum
px=0.5*sprs.block_diag((pxss,-pxss.T)) #x momentum operator in particle-hole space
pyss=fx.sprspy(L)
py=0.5*sprs.block_diag((pyss,-pyss.T))

#evaluation of momentum over different eigenvectors
kx=np.zeros(len(vals))
for j in range(len(vals)):
    kx[j]=fx.expvalue(px,vecs[:,j])

ky=np.zeros(len(vals))
for j in range(len(vals)):
    ky[j]=fx.expvalue(py,vecs[:,j])

#construction of momentum operator in 2D with sparse matrices (already multiplied by -i)
def sprspy(L):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighOBC(i,L)
        matr.appendx(i,nearn[3],-1j)
        matr.attendx(i,nearn[1],1j)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

#tight binding 2D scalable energy
def en(asc,ordin,t):
    return -2*t*(np.cos(asc*a)+np.cos(ordin*a)-2)/a**2


##heat map figure of 2D spectrum
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_xlabel('kx')
ax.set_ylabel('ky')
asc=np.linspace(-np.pi/L,np.pi/L,100)
asc,ordin=np.meshgrid(asc,asc)
ax.contourf(asc,ordin,fx.en(asc,ordin,t,L,0),200)
axp=ax.scatter(kx[:],ky[:],c=vals[:],edgecolors='black')
cb = plt.colorbar(axp)

#spectrum of 2D free ham shape
ax.plot_surface(asc, ordin, fx.en(asc,ordin,t,a,mu0), alpha=0.5) #energy with finite spacing
ax.plot_surface(asc,ordin, t*((asc)**2+(ordin)**2)-mu0, alpha=0.2) #free energy




