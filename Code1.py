# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:30:41 2019

@author: Nicco
"""
from __future__ import division, print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
import functions as fx
from scipy import sparse as sprs
from scipy import linalg as alg


'''
hamiltonian parameters
'''
L=8 #lattice edge 
Ltilde=31 #number of edge sites a matrix element for every lattice point, total matrix  2L^2 x 2L^2
a=L/Ltilde #lattice spacing
t=1./2 #hopping
mu=8 #chemical potential
delta=3 #coupling
ph=np.array([(10.**-5)*1j,(10.**-4)*-1j,(10.**-5)*-1j,(10.**-4)*1j,]) #phase to break degeneracy
vortex=[20,21]
'''
kinetic and pairing hamiltonian building (matrices with O(L^4) elements, only O(L^2) filled), note that
pai10ing is created in distruction sector (lower left). Parts are the stacked to create full hamiltonian
'''
kinet=fx.sprshamOBC(Ltilde,t,-4.*t+a**2*mu,ph,3.88/a) 
pair=fx.nonconsOBC(Ltilde,a*delta/2)
ham=a**-2*sprs.hstack((sprs.vstack((kinet,pair)),sprs.vstack((-(pair.conjugate()),-(kinet.T)))))
'''
diagonalization using either sparse or full matrix diagonalization. 
Full faster for small marices and with it full spectrum obtainable
'''
tic=time.time()
#vals,vecs=sprs.linalg.eigsh(ham,k=2*(Ltilde**2)-2,which='SM') #hamiltonian diagonalization (SM means sorting eigenvalues by smallest modulus)
vals,vecs=alg.eigh(ham.toarray())
toc=time.time()
print("time="+str(toc-tic)) #prints how long the diagonalization process takes
np.savetxt('provals.txt',vals) #saves eigenvalues and eigenvectors on file
np.savetxt('provecs.txt',vecs,fmt='%.18e')

#plot of 3D density profile
asc=np.linspace(0,Ltilde-1,Ltilde,dtype=np.int) #edge of system with correct dimension
asc,ordin=np.meshgrid(asc,asc) #mesh of lattice grid
num=sum(sum(fx.densityplot(asc,ordin,vecs,Ltilde))) #total number of particles
fig=plt.figure(2)
ax=fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('a='+str(a)+', t='+str(t)+', mu='+str(mu)+', delta='+str(delta))
dens=fx.densityplot(asc,ordin,vecs,Ltilde)/(a**2)
ax.scatter(asc*a,ordin*a,dens,'bo') #plot of superfluid density
#ax.scatter(asc,ordin,np.angle(fx.phaseplot(asc,ordin,vortex,Ltilde),deg=True),'bo') #plot of superfluid density
print("num="+str(num))
print(((L**2)/(4*np.pi))*(mu/t))

#plot of energy values on a line to understand if there's a gap
plt.figure(3)
plt.grid(True)
plt.ylim(-10,10)
plt.xlim((len(vals)/2)-6,(len(vals)/2)+6)
plt.plot(np.linspace(0,len(vals),len(vals)),vals,'bo',markersize=1.5)
print(vals[Ltilde**2])

    