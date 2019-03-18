# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:36:48 2019

@author: Nicco
"""
from scipy import sparse as sprs
import numpy as np

class matrixdata: #to build a sparse matrix I need to give 3 array of info (row, column and data)
    def __init__(self): #arrays initialization
        self.row=np.array([],dtype=np.int16) 
        self.col=np.array([],dtype=np.int16)
        self.data=np.array([])
    
    def appendx(self,x,y,dat):
        self.row=np.append(self.row,x)
        self.col=np.append(self.col,y)
        self.data=np.append(self.data,dat)

def nearneigh2D(i,L): #inputs: i=number of the site, L=lattice size
    nearn=np.zeros(4) 
    nearn[0]=int((i+1)%L+L*np.floor(i/L)) #right element
    nearn[1]=int((i+L)%(L**2)) #bottom
    nearn[2]=int((i-1)%L+L*np.floor(i/L)) #left
    nearn[3]=int((i-L)%(L**2)) #top
    return np.int16(nearn)

def sparseham1d(sizeham,t,mu):
    matr=matrixdata()
    for i in range(sizeham):
        matr.appendx(i,i,-mu) #set chem pot on diagonal
        matr.appendx(i,(i+1)%sizeham,-t*np.exp(10.**-4*1.j)) #set hopping left
        matr.appendx(i,(i-1)%sizeham,-t*np.exp(10.**-4*-1.j)) #set hopping right
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def sparsemom1d(sizeham):
    matr=matrixdata()
    for i in range(sizeham):
        matr.appendx(i,i,0)
        matr.appendx(i,(i+1)%sizeham,1j)
        matr.appendx(i,(i-1)%sizeham,-1j)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))
    
def sprsham(L,t,mu,phase): #DANGER mu sign mst be opposite of t
    matr=matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu)
        nearn=nearneigh2D(i,L)
        #phase=np.array([(10.**-5)*1j,(10.**-4)*-1j,(10.**-5)*-1j,(10.**-4)*1j,])
        for j in range(len(nearn)):
            matr.appendx(i,nearn[j],-t*np.exp(phase[j]))    
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def noncons(L,delta):
    row=np.array([],dtype=np.int16)
    col=np.array([],dtype=np.int16)
    data=np.array([])
    for i in range(L**2):
        row=np.append(row,i)
        col=np.append(col,i)
        data=np.append(data,0)
        nearn=nearneigh2D(i,L)
        row=np.append(row,i)
        col=np.append(col,nearn[0])
        data=np.append(data,delta)
        row=np.append(row,i)
        col=np.append(col,nearn[1])
        data=np.append(data,-1j*delta)
        row=np.append(row,i)
        col=np.append(col,nearn[2])
        data=np.append(data,-delta)
        row=np.append(row,i)
        col=np.append(col,nearn[3])
        data=np.append(data,1j*delta)
    return sprs.coo_matrix((data,(row,col)))
        
def expvalue(a,vec):
    exp=np.conj(vec)@a@vec
    return exp

def en(asc,ordin,t,a,mu): # scalable tight binding 2D energy
    return -2*t*(np.cos(asc*a)+np.cos(ordin*a)-2)-mu

def enbog(kx,ky,t,a,mu,delta):
    return np.sqrt((en(kx,ky,t,a,mu))**2+(delta)**2*(kx*a)**2+(ky*a)**2)

