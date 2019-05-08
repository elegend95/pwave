# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:36:48 2019

@author: Nicco
"""
from __future__ import division,print_function
from scipy import sparse as sprs
import numpy as np
import classes as cl
from numba import njit

'''
nearest neighbors with periodic boundary conditons
'''
def nearneighPBC(i,L): #inputs: i=number of the site, L=lattice size
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
def nearneighOBC(i,L): #inputs: i=number of the site, L=lattice size
    nearn=np.zeros(4)
    if ((i+1)%L==0):
        nearn[0]=-1 
    else:
        nearn[0]=i+1 #right element
    if ((i+L)>=(L**2)):
        nearn[1]=-1 
    else:
        nearn[1]=i+L #bottom
    if ((i-1)%L==(L-1)):
        nearn[2]=-1 
    else:
        nearn[2]=i-1 #left
    if ((i-L)<0):
        nearn[3]=-1 
    else:
        nearn[3]=i-L #top

    return np.int16(nearn)

'''
hamiltonian building for periodic boundary conditions
'''

def sprshamPBC(L,t,mu,phase): #hopping hamiltonian
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu) #chemical potential (mu sign must be opposite of t)
        nearn=nearneighPBC(i,L)
        for j in range(len(nearn)):
            matr.appendx(i,nearn[j],-t*np.exp(phase[j])) #hopping
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def sprshamPBCfull(Ltilde,t,mu,phase): #hopping hamiltonian
    matr=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex128)
    for i in range(Ltilde**2):
        matr[i][i]=-mu #chemical potential (mu sign must be opposite of t)
        nearn=nearneighPBC(i,Ltilde)
        for j in range(len(nearn)):
            matr[i][nearn[j]]=-t*np.exp(phase[j])
    return matr

def nonconsPBC(L,delta): #superconducting pairing term
    matr=cl.matrixdata()
    for i in range(L**2):
        nearn=nearneighPBC(i,L)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta]) #px+ipy coupling discretized
        for j in range(len(nearn)):    
            matr.appendx(i,nearn[j],coupl[j])
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def nonconsPBCfull(Ltilde,delta): #superconducting pairing term
    matr=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex128)
    for i in range(Ltilde**2):
        nearn=nearneighPBC(i,Ltilde)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta]) #px+ipy coupling discretized
        for j in range(len(nearn)):    
            matr[i][nearn[j]]=coupl[j]
    return matr

def nonconsPBCvortex(L,delta,vortex): #superconducting pairing with a vortex
    matr=cl.matrixdata()
    for i in range(L**2):
        nearn=nearneighPBC(i,L)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):
            coup=np.sqrt(phaser(i,vortex,L)*phaser(nearn[j],vortex,L)) #vortex is a tuple with vortex coordinates
            matr.appendx(i,nearn[j],kcoup[j]*coup)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))    


'''
hamiltonian building for open boundary conditions
'''

#+confinementwell(L,r,i)

def sprshamOBC(Ltilde,t,mu,phase,r): #hopping hamiltonian
    matr=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex128)
    for i in range(Ltilde**2):
        matr[i][i]=-mu+confinementwell(Ltilde,r,i) #chemical potential (mu sign must be opposite of t)
        nearn=nearneighOBC(i,Ltilde)
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            matr[i][nearn[j]]=-t*np.exp(phase[j])
    return matr

def nonconsOBC(Ltilde,delta):
    matr=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex128)
    for i in range(Ltilde**2):
        nearn=nearneighOBC(i,Ltilde)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)): 
            if (nearn[j]==-1):
                continue
            matr[i][nearn[j]]=coupl[j]
    return matr

def nonconsOBCvortex(Ltilde,delta,vortex1):
    matr=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex128)
    for i in range(Ltilde**2):   
        nearn=nearneighOBC(i,Ltilde)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)): 
            if (nearn[j]==-1):
                continue
            coup=(phaser(i,vortex1,Ltilde)+phaser(nearn[j],vortex1,Ltilde))/2
            matr[i][nearn[j]]=kcoup[j]*coup
    return matr

'''
confining potentials
'''
@njit
def confinementwell(Ltilde,r,i): #insert a hard wall circular potential
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    if (np.sqrt((asc-center)**2+(ordin-center)**2)>r):
        return 10**5
    else:
        return 0

def confinementharm(Ltilde,omega,i): #insert an harmonic circular potential
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    return 0.5*omega**2*((asc-center)**2+(ordin-center)**2)

'''
density operator functions
'''

def density(asc,ordin,vecs,Ltilde): #density of superfluid in point (asc,ordin)
    i=cordtoind(asc,ordin,Ltilde)
    return sum(np.conj(vecs[i+Ltilde**2,(Ltilde**2):])*vecs[i+Ltilde**2,(Ltilde**2):])#if i want just one eigenstate remove 'sum' and put the corresponding site

def densityusingu(asc,ordin,vecs,Ltilde):
    i=cordtoind(asc,ordin,Ltilde)
    return sum(np.conj(vecs[i,:(Ltilde**2)])*vecs[i,:(Ltilde**2)])#if i want just one eigenstate remove 'sum' and put the corresponding site

def densityplot(asc,ordin,vecs,Ltilde): #assigns to a mesh grid the related values of density
    n=np.zeros((Ltilde,Ltilde))
    for i in range(Ltilde):
        for j in range(Ltilde):
            n[i][j]=density(asc[i][j],ordin[i][j],vecs,Ltilde)
    return n

'''
angular momentum operator by lattice discretization DESPISED
'''
def angumomop(Ltilde):
    matr=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex64)
    for i in range(Ltilde**2):
        nearn=nearneighOBC(i,Ltilde)
        ordin,asc=indtocord(i,Ltilde)-(Ltilde-1)/2.
        coupl=-1j*np.array([-ordin,asc,ordin,-asc])
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            matr[i][nearn[j]]=coupl[j] 
    return matr

def angumom(asc,ordin,vecs,Ltilde,lop):
    i=cordtoind(asc,ordin,Ltilde)
    return expvalue(lop.toarray(),vecs[i+Ltilde**2,Ltilde**2:])

def angumomplot(asc,ordin,vecs,Ltilde,lop): #assigns to a mesh grid the related values of density
    n=np.zeros((Ltilde,Ltilde))
    for i in range(Ltilde):
        for j in range(Ltilde):
            n[i][j]=angumom(asc[i][j],ordin[i][j],vecs,Ltilde,lop)
    return n

'''
quasi-angular momentum from rotational invariance (see reference latticeangumom)
'''

def angmo(Ltilde,vec):
    return np.around(np.real(-1j*(2/(np.pi))*np.log(vec[0+Ltilde**2]/vec[(Ltilde-1)+Ltilde**2])))

def angmotot(Ltilde,vecs):
    angtot=0
    for i in range(int((Ltilde**2)/2)):
        ango=angmo(Ltilde,vecs[:,Ltilde**2+i])
        if (ango==-2.0):
            ango=2.0
        print(ango)
        angtot+=ango*sum(np.conj(vecs[(Ltilde**2):,Ltilde**2+i])*vecs[(Ltilde**2):,Ltilde**2+i])
    return angtot

'''
angular momentum in momentum space
'''

@njit
def discretefourier(Ltilde,L): #matrix of trasformation between momentum and position basis
    vecsk=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex64)
    for i in range(Ltilde**2):
        y,x=indtocord(i,Ltilde)
        for k in range(Ltilde**2):
            ky,kx=(2*np.pi/L)*(indtocord(k,Ltilde)-(Ltilde-1)/2)
            vecsk[k,i]=np.exp(-1j*(kx*x+ky*y))
    return 1./(Ltilde)*vecsk

@njit
def angumomk(Ltilde): #metrice of angular momentum in momentum basis
    angmat=np.zeros((Ltilde**2,Ltilde**2),dtype=np.complex64)
    for i in range(Ltilde**2):
        ky,kx=indtocord(i,Ltilde)-(Ltilde-1)/2
        for j in range(Ltilde**2):
            qy,qx=indtocord(j,Ltilde)-(Ltilde-1)/2
            if (qy==ky)&(qx==kx):
                continue
            if (qy==ky):
                angmat[i,j]=qy/(kx-qx)
            if (qx==kx):
                angmat[i,j]=-qx/(ky-qy)
    return -1j*angmat

def angumomktop(Ltilde,L):
    return np.matmul(discretefourier(Ltilde,L).T.conj(),np.matmul(angumomk(Ltilde),discretefourier(Ltilde,L))) 

'''
auxiliary functions
'''

def phaser(i,vortex,Ltilde): #returns the phase of a certain lattice point around vortex centre
    asc,ordin=indtocord(i,Ltilde)
    return ((ordin-vortex[0])+1j*(vortex[1]-asc))/(np.sqrt((asc-vortex[0])**2+(ordin-vortex[1])**2))

def expvalue(a,vec): #expectation value of operators
    exp=np.matmul(np.conj(vec),np.matmul(a,vec))
    return exp

def en(asc,ordin,t,a,mu): #scalable tight binding 2D energy
    return -2*t*a**-2*(np.cos(asc*a)+np.cos(ordin*a)-2)-mu

def enbog(kx,ky,t,a,mu,delta): #bogoliubov dispersion in discretized space with spacing a
    return np.sqrt((en(kx,ky,t,a,mu))**2+(delta/a)**2*((np.sin(kx*a))**2+(np.sin(ky*a))**2))

def enbogcont(kx,ky,t,mu,delta): #bogoliubov dispersion in continuum space
    return np.sqrt((t*(kx**2+ky**2)-mu)**2+(delta)**2*(kx**2+ky**2))

@njit
def indtocord(i,Ltilde): #conversion from lattice site to coordinates
    cord=np.array([0,0])
    cord[0]=np.floor(i/Ltilde)
    cord[1]=i%Ltilde
    return cord

def cordtoind(asc,ordin,Ltilde): #conversion from coordinates to lattice site
    return ordin*Ltilde+asc


