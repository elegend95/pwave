# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:30:41 2019

@author: Nicco
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functions as fx
from scipy import sparse as sprs
from scipy.sparse import linalg

L=3 #sites number
Ltilde=20 #lattice edge
sizeham=L**2 #a matrix element for every lattice point, total matrix  2L^2 x 2L^2
a=Ltilde/L
t=1./2
mu=-4.*t
delta=0

#kinetic and pairing hamiltonian building (matrices with O(L^4) elements, only O(L^2) filled)
phase=np.array([(10.**-5)*1j,(10.**-4)*-1j,(10.**-5)*-1j,(10.**-4)*1j,])
kinet=a**-2*fx.sprsham(L,t,mu,phase)
phase2=np.array([(10.**-8)*-1j,(10.**-7)*1j,(10.**-8)*1j,(10.**-7)*-1j,])
kinet2=a**-2*fx.sprsham(L,t,mu,phase)
pair=fx.noncons(L,delta) #n.b this is the pairing ham in the DESTRUCTION sector (lower left)

#hamiltonian creation (refer to notes for the esplicit form) by stacking blocks
#ham=0.5*sprs.hstack((sprs.vstack((kinet,pair)),sprs.vstack((-(pair.conjugate()),-(kinet2.T)))))
ham=0.5*sprs.block_diag((-(kinet2.T),kinet))
vals,vecs=sprs.linalg.eigsh(ham,k=2*sizeham-2,which='SM') #hamiltonian diagonalization (SM means sorting eigenvalues by smallest modulus)
#vals,vecs=scipy.linalg.eigh(ham.toarray())

kx=np.zeros(len(vals))
for r in range(len(vals)):
    kx[r]=-1j*np.log(vecs[0,r]/vecs[1,r])/a
ky=np.zeros(len(vals))
for r in range(len(vals)):
    ky[r]=-1j*np.log(vecs[L,r]/vecs[0,r])/a

#k=np.sqrt(kx**2+ky**2)
#
#plt.plot(k,vals,'bo')
#plt.grid('True')
#asc=np.linspace(0,3*np.pi/a,100)
#plt.plot(asc,fx.enbog(asc,a,50,t,delta))

#heat map figure
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_xlabel('kx')
ax.set_ylabel('ky')
asc=np.linspace(-np.pi/a,np.pi/a,100)
asc,ordin=np.meshgrid(asc,asc)
ax.contourf(asc,ordin,-2*t*(np.cos(asc*a)+np.cos(ordin*a)-2)/a**2,200)
axp=ax.scatter(kx,ky,c=vals,edgecolors='black')
cb = plt.colorbar(axp)

#3D surface figure
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
asc=np.linspace(-np.pi/a,np.pi/a,100)
asc,ordin=np.meshgrid(asc,asc)
ax.view_init(0, 270)
ax.plot_surface(asc, ordin, fx.en(asc,ordin,t,a), alpha=0.5)
ax.scatter(kx,ky,vals,'bo',s=20,c='r')