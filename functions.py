# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:36:48 2019

@author: Nicco
"""
from __future__ import division,print_function
from scipy import sparse as sprs
import numpy as np
import classes as cl

#matrix structure definition
class matrixdata: #to build a sparse matrix I need to give 3 array of info (row, column and data)
    def __init__(self): #arrays initialization
        self.row=np.array([],dtype=np.int16) 
        self.col=np.array([],dtype=np.int16)
        self.data=np.array([])
    
    def appendx(self,x,y,dat):
        self.row=np.append(self.row,x)
        self.col=np.append(self.col,y)
        self.data=np.append(self.data,dat)

############################################################################################
#nearest neighbors with periodic boundary conditons
def nearneighPBC(i,L): #inputs: i=number of the site, L=lattice size
    nearn=np.zeros(4) 
    nearn[0]=int((i+1)%L+L*np.floor(i/L)) #right element
    nearn[1]=int((i+L)%(L**2)) #bottom
    nearn[2]=int((i-1)%L+L*np.floor(i/L)) #left
    nearn[3]=int((i-L)%(L**2)) #top
    return np.int16(nearn)

#nearest neighbors with open boundary conditions
#this function requires roughly double the time of PBC
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

############################################################################################
#hamiltonian building for periodic boundary conditions
def sprshamPBC(L,t,mu,phase): #hopping hamiltonian
    matr=matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu) #chemical potential (mu sign must be opposite of t)
        nearn=nearneighPBC(i,L)
        for j in range(len(nearn)):
            matr.appendx(i,nearn[j],-t*np.exp(phase[j])) #hopping
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def nonconsPBC(L,delta): #superconducting pairing term (in the distruction sector)
    matr=matrixdata()
    for i in range(L**2):
        nearn=nearneighPBC(i,L)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta]) #px+ipy coupling discretized
        for j in range(len(nearn)):    
            matr.appendx(i,nearn[j],coupl[j])
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))

def nonconsPBCvortex(L,delta,vortex): #superconducting pairing with a vortex
    matr=matrixdata()
    for i in range(L**2):
        nearn=nearneighPBC(i,L)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):
            coup=(phaser(i,vortex,L)+phaser(nearn[j],vortex,L))/2 #vortex is a tuple with vortex coordinates
            matr.appendx(i,nearn[j],kcoup[j]*coup)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)))    

############################################################################################
#hamiltonian building for open boundary conditions and a trapping hard wall potential
def sprshamOBC(L,t,mu,phase,r):
    matr=matrixdata()
    for i in range(L**2):
        matr.appendx(i,i,-mu+confinementwell(L,r,i)) #external potential on every site
        nearn=nearneighOBC(i,L)
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            matr.appendx(i,int(nearn[j]),-t*np.exp(phase[j]))    
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)),dtype=np.complex128)    

def nonconsOBC(L,delta):
    matr=matrixdata()
    for i in range(L**2):
        nearn=nearneighOBC(i,L)
        coupl=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)): 
            if (nearn[j]==-1):
                continue
            matr.appendx(i,nearn[j],coupl[j])
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)),dtype=np.complex128)  

def nonconsOBCvortex(Ltilde,delta,vortex): #superconducting pairing with a vortex
    matr=matrixdata()
    for i in range(Ltilde**2):
        nearn=nearneighOBC(i,Ltilde)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            coup=((phaser(i,vortex,Ltilde)+phaser(nearn[j],vortex,Ltilde))/2.)/np.abs((phaser(i,vortex,Ltilde)+phaser(nearn[j],vortex,Ltilde))/2.) #vortex is a tuple with vortex coordinates
            matr.appendx(i,nearn[j],kcoup[j]*coup)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)),dtype=np.complex128)    

def nonconsOBCvortex2(Ltilde,delta,vortex1,vortex2): #superconducting pairing with two vortices vortex
    matr=cl.matrixdata()
    for i in range(Ltilde**2):
        nearn=nearneighOBC(i,Ltilde)
        kcoup=np.array([delta,-1j*delta,-delta,1j*delta])
        for j in range(len(nearn)):
            if (nearn[j]==-1):
                continue
            coup=((phaser(i,vortex1,Ltilde)*phaser(i,vortex2,Ltilde)+phaser(nearn[j],vortex1,Ltilde)*phaser(nearn[j],vortex2,Ltilde))/2.)/(np.abs((phaser(i,vortex1,Ltilde)*phaser(i,vortex2,Ltilde)+phaser(nearn[j],vortex1,Ltilde)*phaser(nearn[j],vortex2,Ltilde))/2.)) #vortex is a tuple with vortex coordinates
            matr.appendx(i,nearn[j],kcoup[j]*coup)
    return sprs.coo_matrix((matr.data,(matr.row,matr.col)),dtype=np.complex128)    

############################################################################################
#confining potentials
def confinementwell(Ltilde,r,i): #insert a hard wall circular potential
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    if (np.sqrt((asc-center)**2+(ordin-center)**2)>r):
        return 10**5
    #elif (np.sqrt((asc-center)**2+(ordin-center)**2)<rmin):
    #    return 10**5
    else:
        return 0

def confinementharm(Ltilde,omega,i): #insert an harmonic circular potential
    center=0.5*(Ltilde-1)
    asc,ordin=indtocord(i,Ltilde)
    return 0.5*omega**2*((asc-center)**2+(ordin-center)**2)

############################################################################################
#density operator functions
def density(asc,ordin,vecs,Ltilde): #density of superfluid in point (asc,ordin)
    i=cordtoind(asc,ordin,Ltilde)
    #return np.imag(vecs[i,2])
    return np.vdot(vecs[i+Ltilde**2,(Ltilde**2):],vecs[i+Ltilde**2,(Ltilde**2):])#if i want just one eigenstate remove 'sum' and put the corresponding site

def densityplot(asc,ordin,vecs,Ltilde): #assigns to a mesh grid the related values of density
    n=np.zeros((Ltilde,Ltilde))
    for i in range(Ltilde):
        for j in range(Ltilde):
            n[i,j]=density(asc[i,j],ordin[i,j],vecs,Ltilde)
    return n


############################################################################################
#auxiliary functions
def phaser(i,vortex,Ltilde): #returns the phase of a certain lattice point around vortex centre
    ordin,asc=indtocord(i,Ltilde)
    return ((asc-vortex[1])+1j*(vortex[0]-ordin))/(np.sqrt((asc-vortex[1])**2+(ordin-vortex[0])**2))

def expvalue(a,vec): #expectation value of operators
    exp=np.vdot(vec,np.matmul(a,vec))
    return np.real(exp)

def en(asc,ordin,t,a,mu): #scalable tight binding 2D energy
    return -2*t*a**-2*(np.cos(asc*a)+np.cos(ordin*a)-2)-mu

def enbog(kx,ky,t,a,mu,delta): #bogoliubov dispersion in discretized space with spacing a
    return np.sqrt((en(kx,ky,t,a,mu))**2+(delta/a)**2*((np.sin(kx*a))**2+(np.sin(ky*a))**2))

def enbogcont(kx,ky,t,mu,delta): #bogoliubov dispersion in continuum space
    return np.sqrt((t*(kx**2+ky**2)-mu)**2+(delta)**2*(kx**2+ky**2))

def indtocord(i,Ltilde): #conversion from lattice site to coordinates
    cord=np.array([0,0])
    cord[0]=np.floor(i/Ltilde)
    cord[1]=i%Ltilde
    return cord

def cordtoind(asc,ordin,Ltilde): #conversion from coordinates to lattice site
    return ordin*Ltilde+asc

def reordin(tri,orto,Ltilde):
    for i in range(Ltilde**2):
        if tri[2*i,2*i+1]<0:
           if np.abs(tri[2*i,2*i+1])>10**-6: 
               tri[2*i:2*i+2,:]=np.flip(tri[2*i:2*i+2,:],axis=0)
               tri[:,2*i:2*i+2]=np.flip(tri[:,2*i:2*i+2],axis=1)
               orto[:,2*i:2*i+2]=np.flip(orto[:,2*i:2*i+2],axis=1)
    return tri,orto

def densityaround(vortex,dens,Ltilde,r):#density profile around a vortex plotted in function of the radius
    densv=np.array([])
    radi=np.array([])
    for i in range(Ltilde**2):
        ordin,asc=indtocord(i,Ltilde)
        rad=np.sqrt((ordin-vortex[0])**2+(asc-vortex[1])**2)
        if rad<r:
            densv=np.append(densv,dens[ordin,asc])
            radi=np.append(radi,rad)
    return radi,densv

    




