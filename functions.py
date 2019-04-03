# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:36:48 2019

@author: Nicco
"""
from scipy import sparse as sprs
import numpy as np
import classes as cl

#nearest neighbors with periodic boundary conditons
def nearneighPBC(i,L): #inputs: i=number of the site, L=lattice size
    nearn=np.zeros(4) 
    nearn[0]=int((i+1)%L+L*np.floor(i/L)) #right element
    nearn[1]=int((i+L)%(L**2)) #bottom
    nearn[2]=int((i-1)%L+L*np.floor(i/L)) #left
    nearn[3]=int((i-L)%(L**2)) #top
    return np.int16(nearn)

#nearest neighbors with open boundary conditions
#this function requires double the time of PBC
def nearneighOBC(i,L): #inputs: i=number of the site, L=lattice size
    nearn=np.array([])
    if ((i+1)%L==0):
        nearn=np.append(nearn,-1) #right element
    else:
        nearn=np.append(nearn,i+1)
    if ((i+L)>=(L**2)):
        nearn=np.append(nearn,-1) #right element
    else:
        nearn=np.append(nearn,i+L)
    if ((i-1)%L==(L-1)):
        nearn=np.append(nearn,-1) #right element
    else:
        nearn=np.append(nearn,i-1)
    if ((i-L)<0):
        nearn=np.append(nearn,-1) #right element
    else:
        nearn=np.append(nearn,i-L)

    return np.int16(nearn)

'''
functions defined for periodic boundary conditions
'''
def sprshamPBC(L,t,mu,phase): #DANGER mu sign mst be opposite of t
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu)
        nearn=nearneighPBC(i,L)
        for j in range(len(nearn)):
            matr.appendx(i,nearn[j],-t*np.exp(phase[j]))    
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def nonconsPBC(L,delta):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighPBC(i,L)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):    
            matr.appendx(i,nearn[j],coupl[j])
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))
   
def nonconsPBCvortex(L,delta,vortex):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighPBC(i,L)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):
            coup=np.sqrt(phaser(i,vortex,L)*phaser(nearn[j],vortex,L))
            matr.appendx(i,nearn[j],kcoup[j]*coup)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))    

'''
functions defined for open boundary conditions
'''

def sprshamOBC(L,t,mu,phase,r):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu)
        nearn=nearneighOBC(i,L)
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            matr.appendx(i,int(nearn[j]),-t*np.exp(phase[j]))    
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))    

#+confinementwell(L,r,i)        

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

def confinementwell(Ltilde,r,i): #insert a hard wall circular potential
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    if (np.sqrt((asc-center)**2+(ordin-center)**2)>r):
        return 10**4
    else:
        return 0

def confinementharm(Ltilde,omega,i):
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    return 0.5*omega**2*((asc-center)**2+(ordin-center)**2)


def number(vecs,Ltilde):
    num=0
    for i in range(Ltilde**2):
        num+=np.conj(vecs[Ltilde**2:,i])@vecs[Ltilde**2:,i]
    return num

def phaser(i,vortex,Ltilde):
    asc,ordin=indtocord(i,Ltilde)
    if (asc==vortex[0])&(ordin==vortex[1]):
        return 0
    return ((ordin-vortex[0])+1j*(vortex[1]-asc))/(np.sqrt((asc-vortex[0])**2+(ordin-vortex[1])**2))

def phasetemp(asc,ordin,vortex,Ltilde):
    if (asc==vortex[0])&(ordin==vortex[1]):
        return 0
    return ((ordin-vortex[0])+1j*(vortex[1]-asc))/(np.sqrt((asc-vortex[0])**2+(ordin-vortex[1])**2))

def density(asc,ordin,vecs,Ltilde):
    i=cordtoind(asc,ordin,Ltilde)
    return sum(np.conj(vecs[i+Ltilde**2,Ltilde**2:])*vecs[i+Ltilde**2,Ltilde**2:])

def densityplot(asc,ordin,vecs,Ltilde):
    n=np.zeros((Ltilde,Ltilde))
    for i in range(Ltilde):
        for j in range(Ltilde):
            n[i][j]=density(asc[i][j],ordin[i][j],vecs,Ltilde)
    return n

def phaseplot(asc,ordin,vortex,Ltilde):
    n=np.zeros((Ltilde,Ltilde),dtype=np.complex)
    for i in range(Ltilde):
        for j in range(Ltilde):
            n[i][j]=phasetemp(asc[i][j],ordin[i][j],vortex,Ltilde)
    return n    

def expvalue(a,vec): #expectation value of operators
    exp=np.conj(vec)@a@vec
    return exp

def en(asc,ordin,t,a,mu): #scalable tight binding 2D energy
    return -2*t*a**-2*(np.cos(asc*a)+np.cos(ordin*a)-2)-mu

def enbog(kx,ky,t,a,mu,delta): #bogoliubov dispersion in discretized space with spacing a
    return np.sqrt((en(kx,ky,t,a,mu))**2+(delta/a)**2*((np.sin(kx*a))**2+(np.sin(ky*a))**2))

def enbogcont(kx,ky,t,mu,delta): #bogoliubov dispersion in continuum space
    return np.sqrt((t*(kx**2+ky**2)-mu)**2+(delta)**2*(kx**2+ky**2))

def indtocord(i,Ltilde):
    cord=np.array([0,0])
    cord[0]=np.floor(i/Ltilde)
    cord[1]=i%Ltilde
    return cord

def cordtoind(asc,ordin,Ltilde):
    return ordin*Ltilde+asc


