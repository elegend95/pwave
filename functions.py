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
    
def sprshamOBC(L,t,mu,phase):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu)
        nearn=nearneighOBC(i,L)
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            matr.appendx(i,int(nearn[j]),-t*np.exp(phase[j]))    
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))    

def sprshamPBC(L,t,mu,phase): #DANGER mu sign mst be opposite of t
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu)
        nearn=nearneighPBC(i,L)
        for j in range(len(nearn)):
            matr.appendx(i,nearn[j],-t*np.exp(phase[j]))    
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def sprspy(L): #py operator matrix (already multiplied by -i)
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighOBC(i,L)
        if (nearn[3]!=-1):    
            matr.appendx(i,nearn[3],-1j)
        if (nearn[1]!=-1):    
            matr.appendx(i,nearn[1],1j)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def sprspx(L): #px operator matrix (already multiplied by -i)
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighOBC(i,L)
        if (nearn[0]!=-1):
            matr.appendx(i,nearn[0],-1j) 
        if (nearn[2]!=-1):            
            matr.appendx(i,nearn[2],1j)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def noncons(L,delta):
    matr=cl.matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,0)
        nearn=nearneighPBC(i,L)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):    
            matr.appendx(i,nearn[j],coupl[j])
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))
        
def expvalue(a,vec):
    exp=np.conj(vec)@a@vec
    return exp

def en(asc,ordin,t,a,mu): # scalable tight binding 2D energy
    return -2*t*a**-2*(np.cos(asc*a)+np.cos(ordin*a)-2)-mu

def enbog(kx,ky,t,a,mu,delta):
    return np.sqrt((en(kx,ky,t,a,mu))**2+(delta/a)**2*((np.sin(kx*a))**2+(np.sin(ky*a))**2))

def enbogcont(kx,ky,t,mu,delta):
    return np.sqrt((t*(kx**2+ky**2)-mu)**2+(delta)**2*(kx**2+ky**2))
