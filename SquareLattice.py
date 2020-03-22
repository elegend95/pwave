# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:30:41 2019

@author: niccobal
"""
from __future__ import division, print_function
import time
import numpy as np
import Functions as fx

#plotting packages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#algebra packages
import scipy.sparse as sprs
from scipy import linalg as alg
 
############################################################################################
#hamiltonian parameters
L=4. #lattice edge 
Ltilde=20   #number of edge sites a matrix element for every lattice point, total matrix  2L^2 x 2L^2
a=L/(Ltilde-1) #lattice spacing
R=2

t=1 #hopping
mu=8. #chemical potential
delta=0. #coupling
ph=np.array([(10.**-10)*1j,(10.**-9)*-1j,(10.**-10)*-1j,(10.**-9)*1j,]) #phase to break degeneracy

#vortices coordinates (remember that in our convention the first number is the ordinate)
#vortex1=[4.000000000001/a,2.000000000002/a]
#vortex2=[4.0000000000002/a,2.000000002/a]

############################################################################################
#kinetic and pairing hamiltonian building (matrices with O(L^4) elements, only O(L^2) filled)

kinet=fx.sprshamOBC(Ltilde,t,-4.*t+a**2*mu,ph,R/a) 
pair=fx.nonconsOBC(Ltilde,a*delta/2.)
ham=a**-2*sprs.hstack((sprs.vstack((kinet,pair)),sprs.vstack((-(pair.conjugate()),-(kinet.T))))) #pairing is created in distruction sector (lower left). Parts are then stacked to create full hamiltonian

tic=time.time()
vals,vecs=alg.eigh(ham.toarray()) #diagonalization using either full matrix diagonalization. 
toc=time.time()
print("time="+str(toc-tic)) #prints how long the diagonalization process takes

############################################################################################
#data saving in a compressed format, np.load() needed to read the data
'''
np.savez_compressed('../plots/21maggio/provals'+str(Ltilde),vals,fmt='%.9f')
np.savez_compressed('../plots/21maggio/provecs'+str(Ltilde),vecs,fmt='%.9e') #9 decimals to save up some space
'''
############################################################################################
#heat plot of density profile
asc=np.linspace(0,Ltilde-1,Ltilde,dtype=np.int) #edge of system with correct dimension
asc,ordin=np.meshgrid(asc,asc) #mesh of lattice grid
dens=fx.densityplot(asc,ordin,vecs,Ltilde)/(a**2) #density in each point (scaled)

fig=plt.figure(Ltilde+4,figsize=(10,10)) #initialization of plot
ax2=fig.add_subplot(111) #plot inside figure
ax2.set_xlabel('x')
ax2.set_ylabel('y')
axp=ax2.contourf(asc*a,ordin*a,dens,500) #contour plot of density

divider = make_axes_locatable(ax2) #setup of the colorbar on the top (next three lines)
cax = divider.append_axes('top',size="5%", pad=0.05)
plt.colorbar(axp,orientation='horizontal', cax=cax,ticks=np.linspace(0,np.max(dens),5))
cax.xaxis.set_ticks_position("top")
plt.tight_layout()
'''
############################################################################################
#plot of energy spectrum around zero
plt.figure(3)
plt.grid(True)
plt.ylim(-5.5,5.5)
plt.ylabel('Energy')
plt.xlim((len(vals)/2)-40,(len(vals)/2)+40)
plt.xticks([], [])

plt.plot(np.linspace(0,len(vals),len(vals)),vals,'ro') #plot of energies
plt.tight_layout()

def densityaround(dens,Ltilde,r):#density profile around a vortex plotted in function of the radius
    densv=np.array([])
    radi=np.array([])
    init=int(Ltilde*(Ltilde+1)/2)
    fin=int(Ltilde/2+Ltilde*(Ltilde+1)/2)
    for i in range(init,fin,1):
        ordin,asc=fx.indtocord(i,Ltilde)
        densv=np.append(densv,dens[ordin,asc])
        radi=np.append(radi,asc-Ltilde/2)
    return radi,densv

plt.figure(70)
plt.grid(True)
plt.plot(densityaround(dens,Ltilde,R)[0]*a,densityaround(dens,Ltilde,R)[1])
plt.tight_layout()
'''
