# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:23:34 2019

@author: Nicco
"""

from __future__ import division,print_function
from scipy import sparse as sprs
import numpy as np
import classes as cl
from libc.math import sqrt,floor


cdef double csqrt(double x):
    return sqrt(x)

cdef int cfloor(double x):
    return floor(x)

'''
nearest neighbors with periodic boundary conditons
'''
cdef int[:] nearneighPBC(int i,int L): #inputs: i=number of the site, L=lattice size
    cdef int[:] nearn
    nearn=np.zeros(4) 
    nearn[0]=int((i+1)%L+L*np.floor(i/L)) #right element
    nearn[1]=int((i+L)%(L**2)) #bottom
    nearn[2]=int((i-1)%L+L*np.floor(i/L)) #left
    nearn[3]=int((i-L)%(L**2)) #top
    return np.int16(nearn)

'''
nearest neighbors with open boundary conditions
this function requires roughly double the time of PBC
'''
cdef nearneighOBC(int i,int L): #inputs: i=number of the site, L=lattice size
    nearn=np.array([])
    if ((i+1)%L==0):
        nearn=np.append(nearn,-1) 
    else:
        nearn=np.append(nearn,i+1) #right element
    if ((i+L)>=(L**2)):
        nearn=np.append(nearn,-1) 
    else:
        nearn=np.append(nearn,i+L) #bottom
    if ((i-1)%L==(L-1)):
        nearn=np.append(nearn,-1) 
    else:
        nearn=np.append(nearn,i-1) #left
    if ((i-L)<0):
        nearn=np.append(nearn,-1) 
    else:
        nearn=np.append(nearn,i-L) #top

    return np.int16(nearn)

'''
functions defined for periodic boundary conditions
'''
cdef sprshamPBC(int L,double t,double mu,complex[:] phase): #hopping hamiltonian
    cdef int i
    cdef int [:] nearn
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu) #chemical potential (mu sign must be opposite of t)
        nearn=nearneighPBC(i,L)
        for j in range(len(nearn)):
            matr.appendx(i,nearn[j],-t*np.exp(phase[j])) #hopping
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def nonconsPBC(L,delta): #superconducting pairing term
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighPBC(i,L)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta]) #px+ipy coupling discretized
        for j in range(len(nearn)):    
            matr.appendx(i,nearn[j],coupl[j])
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))
   
def nonconsPBCvortex(L,delta,vortex): #superconducting pairing with a vortex
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighPBC(i,L)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):
            coup=np.sqrt(phaser(i,vortex,L)*phaser(nearn[j],vortex,L)) #vortex is a tuple with vortex coordinates
            matr.appendx(i,nearn[j],kcoup[j]*coup)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))    

'''
functions defined for open boundary conditions
'''

cdef sprshamOBC(int L, t, mu,phase,r):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu+confinementwell(L,r,i)) #external potential on every site
        nearn=nearneighOBC(i,L)
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            matr.appendx(i,int(nearn[j]),-t*np.exp(phase[j]))    
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))    

def nonconsOBC(L,delta):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighOBC(i,L)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)): 
            if (nearn[j]==-1):
                continue
            matr.appendx(i,nearn[j],coupl[j])
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))  

def nonconsOBCvortex(L,delta,vortex):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighOBC(i,L)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)): 
            if (nearn[j]==-1):
                continue
            coup=np.sqrt(phaser(i,vortex,L)*phaser(nearn[j],vortex,L))
            matr.appendx(i,nearn[j],kcoup[j]*coup)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

'''
confining potentials
'''

cdef double confinementwell(int Ltilde,double r,int i): #insert a hard wall circular potential
    cdef double center
    cdef int asc
    cdef int ordin
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    if (csqrt((asc-center)**2+(ordin-center)**2)>r):
        return 10**4
    else:
        return 0

def confinementharm(Ltilde,omega,i): #insert an harmonic circular potential
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    return 0.5*omega**2*((asc-center)**2+(ordin-center)**2)

'''
density operator functions
'''

def phaser(i,vortex,Ltilde): #returns the phase of a certain lattice point around vortex centre
    asc,ordin=indtocord(i,Ltilde)
    if (asc==vortex[0])&(ordin==vortex[1]):
        return 0
    return ((ordin-vortex[0])+1j*(vortex[1]-asc))/(np.sqrt((asc-vortex[0])**2+(ordin-vortex[1])**2))

def density(asc,ordin,vecs,Ltilde): #density of superfluid in point (asc,ordin)
    i=cordtoind(asc,ordin,Ltilde)
    return np.conj(vecs[i+Ltilde**2,Ltilde**2])*vecs[i+Ltilde**2,Ltilde**2]

def densityplot(asc,ordin,vecs,Ltilde): #assigns to a mesh grid the related values of density
    n=np.zeros((Ltilde,Ltilde))
    for i in range(Ltilde):
        for j in range(Ltilde):
            n[i][j]=density(asc[i][j],ordin[i][j],vecs,Ltilde)
    return n

'''
auxiliary functions
'''

def expvalue(a,vec): #expectation value of operators
    exp=np.matmul(np.conj(vec),np.matmul(a,vec))
    return exp

def en(asc,ordin,t,a,mu): #scalable tight binding 2D energy
    return -2*t*a**-2*(np.cos(asc*a)+np.cos(ordin*a)-2)-mu

def enbog(kx,ky,t,a,mu,delta): #bogoliubov dispersion in discretized space with spacing a
    return np.sqrt((en(kx,ky,t,a,mu))**2+(delta/a)**2*((np.sin(kx*a))**2+(np.sin(ky*a))**2))

def enbogcont(kx,ky,t,mu,delta): #bogoliubov dispersion in continuum space
    return np.sqrt((t*(kx**2+ky**2)-mu)**2+(delta)**2*(kx**2+ky**2))

cdef int[:] indtocord(int i,int Ltilde): #conversion from lattice site to coordinates
    cdef int cord[2]
    cord[0]=floor(i/Ltilde)
    cord[1]=i%Ltilde
    return cord

cdef int cordtoind(int asc,int ordin,int Ltilde): #conversion from coordinates to lattice site
    return ordin*Ltilde+asc
