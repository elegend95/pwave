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
import sys

'''
hamiltonian parameters
'''
L=8 #lattice edge 
Ltilde=40 #number of edge sites a matrix element for every lattice point, total matrix  2L^2 x 2L^2
a=L/(Ltilde-1) #lattice spacing
t=1./2 #hopping
mu=-30 #chemical potential
delta=10 #coupling
ph=np.array([(10.**-5)*1j,(10.**-4)*-1j,(10.**-5)*-1j,(10.**-4)*1j,]) #phase to break degeneracy

'''
vortices coordinates
'''

vortex1=[2.51/a,2.51/a]
#vortex2=[5.75/a,5.75/a]
#vortex3=[4.02/a,4.01/a]

'''
kinetic and pairing hamiltonian building (matrices with O(L^4) elements, only O(L^2) filled), note that
pai10ing is created in distruction sector (lower left). Parts are the stacked to create full hamiltonian
'''

tic=time.time()
kinet=fx.sprshamOBC(Ltilde,t,-4.*t+a**2*mu,ph,3.88/a) 
pair=fx.nonconsOBC(Ltilde,a*delta/2)
ham=a**-2*np.hstack((np.vstack((kinet,pair)),np.vstack((-(pair.conjugate()),-(kinet.T)))))

'''
diagonalization using either sparse or full matrix diagonalization. 
Full faster for small marices and with it full spectrum obtainable
'''   
#vals,vecs=sprs.linalg.eigsh(ham,k=2*(Ltilde**2)-2,which='SM') #hamiltonian diagonalization (SM means sorting eigenvalues by smallest modulus)
vals,vecs=alg.eigh(ham)
toc=time.time()
print("time="+str(toc-tic)) #prints how long the diagonalization process takes

'''
data saving in a compressed format, np.load() needed to read the data
'''
'''
np.savez_compressed('provals'+str(Ltilde),vals,fmt='%.9f')
np.savez_compressed('provecs'+str(Ltilde),vecs,fmt='%.9e') #9 decimals to save up some space
'''

    
