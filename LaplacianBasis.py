# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:03:43 2020

@author: niccobal
"""

from __future__ import division
from scipy import special as sp
from scipy import integrate as integ
import numpy as np
import matplotlib.pyplot as plt
import Functions as fx
from numba import njit,jit
from scipy import linalg as alg
import time

#basis function parameters
R=2 #radius of disk
N=5 #maximum number of bessel zeros (index n in hamiltonian)
M=5 #maximum order of bessel function (index m in hamiltonian)

#hamiltonian parameters
T=1
MU=8
D=0.

############################################################################################
#definition of basis functions
@jit
def radial(r,rmax,m,n): #normalized basis of radial eigenfunctions of laplacian
    m=np.abs(m)
    N=(rmax**2/2)*(sp.jv(m+1,sp.jn_zeros(m,n)[n-1]))**2 #normalization
    return (1./np.sqrt(N))*sp.jv(np.abs(m),r*sp.jn_zeros(m,n)[n-1]/rmax)
    
@jit
def basisfunc(r,phi,rmax,m,n):  #basis of laplacian eigenfunctions constructed by multiplying radial part by phase
    return radial(r,rmax,m,n)*np.exp(1j*m*phi)*(1./np.sqrt(2*np.pi)) #n.b. both positive and negative m required

def basislattice(Ltilde,rmax,m,n): #recasting of the basis function on a square lattice
    points=np.zeros(Ltilde**2,dtype=np.complex128)
    for i in range(Ltilde**2):
        y,x=(fx.indtocord(i,Ltilde)-(Ltilde-1)/2)/(0.5*Ltilde/rmax) #normalization of coordinates in the max radius
        points[i]=basisfunc(np.sqrt(y**2+x**2),np.arctan2(y,x),rmax,m,n)
        print(i,x,y)
        if (np.sqrt(y**2+x**2)>rmax):
            points[i]=0
    return points

############################################################################################
#construction of diagonal matrix for the kinetic part
zerbes=np.zeros((N*(2*M-1),3)) #array for the zeros of bessel functions with corresponding quantum numbers
zerbes[0:N,0]=sp.jn_zeros(0,N) #data in each row is (energy,m,n)
for i in range(N):
    zerbes[i,1]=0
    zerbes[i,2]=i
for i in range(1,M):
    zerbes[N*(2*i-1):N*(2*i),0]=sp.jn_zeros(i,N)
    for j in range(N):
        zerbes[N*(2*i-1)+j,1]=i
        zerbes[N*(2*i-1)+j,2]=j
    zerbes[N*(2*i):N*(2*i+1),0]=sp.jn_zeros(i,N)
    for j in range(N):
        zerbes[N*(2*i)+j,1]=-i
        zerbes[N*(2*i)+j,2]=j
        
zerbes=zerbes[np.lexsort((zerbes[:,1], zerbes[:,2]))] #sorting of indices in order to put all the elemts with equal n together
eigvals=((T*(zerbes/R)**2)-MU) #computation of eigenvalues for each pair (m,n)
kinet=np.diag(zerbes[:,0]) #diagonal kinetic energy matrix  

vals,vecs=alg.eigh(kinet)    

#recasting eigenfunctions in a position basis for plotting purposes
Ltilde=35 #number of lattice points to plot
posiz=np.zeros((Ltilde*Ltilde,(2*M-1)*N),dtype=np.complex128)



basislatticematrix=np.zeros((Ltilde*Ltilde,(2*M-1)*N),dtype=np.complex128)
for k in range((2*M-1)*N): #DO NOT RUN THESE LINES IF NOT NEEDED
    basislatticematrix[:,k]=basislattice(Ltilde,R,int(zerbes[k,1]),int(zerbes[k,2]+1))

posiz=np.matmul(vecs.T,basislatticematrix.T)
posiz=posiz.T
#posiz=basislattice(Ltilde,R,0,1)

punti=np.linspace(0,(Ltilde**2)-1,(Ltilde**2),dtype=np.int64) #initialization of a square grid to be recast as radial
y=(np.floor(punti/Ltilde)-(Ltilde-1)/2)/(Ltilde/(2*R))
x=(punti%Ltilde-(Ltilde-1)/2)/(Ltilde/(2*R))


dens=np.zeros(Ltilde**2) #density of superfluid 
for i in range(Ltilde**2):
    dens[i]=sum((posiz[i,:].conj()*posiz[i,:]))
#dens=posiz[:].conj()*posiz[:] #to use only when diagonalizing just free hamiltonian

#figure: density of superfluid
fig=plt.figure(4,figsize=(10,10))
fig.add_subplot(111,aspect='equal')

plt.scatter(x,y,c=dens,marker='s',s=180) #scatter plot of density

plt.ylim(-R,R)
plt.xlim(-R,R)
plt.colorbar()

'''
############################################################################################
#construction of coupling term
#coupl=np.zeros((N*(2*M-1),N*(2*M-1)))
def integrand(r,rmax,m,n,nu):
    return r*sp.jv(m+1,r*sp.jn_zeros(m,nu)[nu-1]/rmax)*sp.jv(m+1,r*sp.jn_zeros(m,n)[n-1]/rmax)

coupln=np.empty((N,2*M-1,2*M-1))
for n in range(N):
    matr=np.zeros((2*M-1,2*M-1))
    tic=time.time()
    print(n)
    for j in range(M):
        rmax=1
        m=indices[j][0]
        ni=n+1
        mp=-m-1
        matr[j,2*(M-1)-1-j]=(-1.)**m*(sp.jn_zeros(m,ni)[ni-1]/rmax)*integ.quad(integrand,0,rmax,args=(rmax,m,ni,ni))[0]*\
            (1./(np.sqrt(((rmax**2/2)*(sp.jv(m+1,sp.jn_zeros(m,ni)[n-1]))**2)*((rmax**2/2)*(sp.jv(m+2,sp.jn_zeros(m+1,ni)[ni-1]))**2)))) 
    coupln[n]=matr
    toc=time.time()
    print(tic-toc)

coupl=coupln[0]
for i in range(1,N):
    coupl=alg.block_diag(coupl,coupln[n])
    
coupl=D*(coupl-coupl.T)
ham=np.hstack((np.vstack((kinet,coupl)),np.vstack((-(coupl.conj()),-(kinet.T))))) #construction by stacking

tic=time.time()
print(integ.quad(integrand,0,rmax,args=(rmax,1,2,2))[0])
toc=time.time()
print(toc-tic)
    
plt.figure(3)
plt.grid(True)
plt.ylim(-50.5,50.5)
plt.ylabel('Energy')    
plt.xlim((len(vals)/2)-40,(len(vals)/2)+30)
plt.xticks([], [])

plt.plot(np.linspace(0,len(vals),len(vals)),vals,'bo') #plot of energies
plt.tight_layout()
'''

    
    
    
    