# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:30:41 2019

@author: Nicco
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
import functions as fx
from scipy import sparse as sprs
from scipy import linalg as alg

L=1 #lattice edge 
Ltilde=3 #number of edge ites
sizeham=Ltilde**2 #a matrix element for every lattice point, total matrix  2L^2 x 2L^2
a=L/Ltilde #lattice spacing
t=1./2 #hoppin2
mu0=0 #chemical potential
mu=-4.*t+a**2*mu0 #discretized renormalized impulse
delta=0 #coupling
phase=np.array([(10.**-5)*1j,(10.**-4)*-1j,(10.**-5)*-1j,(10.**-4)*1j,]) #phase to break degeneracy

#kinetic and pairing hamiltonian building (matrices with O(L^4) elements, only O(L^2) filled)
kinet=fx.sprshamPBC(Ltilde,t,mu,phase) 
pair=fx.noncons(Ltilde,a*delta/2) #n.b this is the pairing ham in the DESTRUCTION sector (lower left), pairing is modified by discretization
    
#hamiltonian creation (refer to notes for the esplicit form) by stacking blocks
ham=a**-2*sprs.hstack((sprs.vstack((kinet,pair)),sprs.vstack((-(pair.conjugate()),-(kinet.T)  ))))
tic=time.time()
#vals,vecs=sprs.linalg.eigsh(ham,k=2*sizeham-2,which='SM') #hamiltonian diagonalization (SM means sorting eigenvalues by smallest modulus)
vals,vecs=alg.eigh(ham.toarray())
toc=time.time()
print(toc-tic) #prints how long the diagonalization process takes


kx=np.zeros(len(vals))
ky=np.zeros(len(vals))
for r in range(len(vals)):
    kx[r]=-1j*np.log(vecs[0,r]/vecs[1,r])/a
    ky[r]=-1j*np.log(vecs[Ltilde,r]/vecs[0,r])/a
    if (np.abs(vecs[0,r])<10**-8)&(np.abs(vecs[1,r])<10**-8): #correct behaviour when kx and ky should be zero (bit of workaround)
        kx[r]=-1j*np.log((vecs[Ltilde**2,r])/(vecs[1+Ltilde**2,r]))/a
    if (np.abs(vecs[Ltilde,r])<10**-8)&(np.abs(vecs[0,r])<10**-8):
        ky[r]=-1j*np.log((vecs[Ltilde+Ltilde**2,r])/(vecs[Ltilde**2,r]))/a
'''
#construction of momentum operator, gives problem bc we have to do arcsin at the end and there is doubling problem
pxss=a**-2*fx.sprspx(Ltilde) #construction of upper block of x direction momentum
px=0.5*sprs.block_diag((pxss,-pxss.T)) #x momentum operator in particle-hole space
pyss=a**-2* fx.sprspy(Ltilde)
py=0.5*sprs.block_diag((pyss,-pyss.T))

#evaluation of momentum over different eigenvectors
qx=np.zeros(len(vals))
for j in range(len(vals)):
    qx[j]=fx.expvalue(pxss,vecs[:,j])

qy=np.zeros(len(vals))
for j in range(len(vals)):
    qy[j]=fx.expvalue(pyss,vecs[:,j])

kx=np.arcsin(qx*a)/a
ky=np.arcsin(qy*a)/a
'''
#3D surface figure, setting parameters
asc=np.linspace(-np.pi/a,np.pi/a,100)
asc,ordin=np.meshgrid(asc,asc)    
fig=plt.figure(1)
ax=fig.add_subplot(111, projection='3d')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_title('a='+str(a)+', t='+str(t)+', mu='+str(mu0)+', delta='+str(delta))
ax.view_init(0, 90)
ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))

#theoretical bogoliubov dispersion
ax.plot_surface(asc, ordin, fx.enbog(asc,ordin,t,a,mu0,delta), alpha=0.2,color='b')
ax.plot_surface(asc, ordin, -fx.enbog(asc,ordin,t,a,mu0,delta), alpha=0.2,color='b')
#calculated eigenvalu
ax.scatter(kx,ky,vals,'bo',s=20,c='r')
  
#continuum limit  
#ax.plot_surface(asc, ordin, fx.en(asc,ordin,t,a,mu0), alpha=0.5) #energy with finite spacing
#ax.plot_surface(asc, ordin, fx.enbogcont(asc,ordin,t,mu0,delta), alpha=0.2,color='red')
plt.show()

#heat plot to understand outliers
kpos=np.array([])
for i in range(len(vals)):
    if (vals[i]>=0):
        kpos=np.append(kpos,i)
        
        
'''
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Ltilde='+str(Ltilde)+', t='+str(t)+', mu='+str(mu0)+', delta='+str(delta))
ax.set_xlabel('kx')
ax.set_ylabel('ky')
asc=np.linspace(-np.pi/a,np.pi/a,100)
asc,ordin=np.meshgrid(asc,asc)
mezzival=Ltilde**2
ax.set_title('a='+str(a)+', t='+str(t)+', mu='+str(mu0)+', delta='+str(delta))
mi = np.min((vals[mezzival:].min(), fx.enbog(asc,ordin,t,a,mu0,delta).min()))
ma = np.max((vals[mezzival:].max(), fx.enbog(asc,ordin,t,a,mu0,delta).max()))
norm = clr.Normalize(vmin=mi,vmax=ma)
ax.contourf(asc,ordin,fx.enbog(asc,ordin,t,a,mu0,delta),200,norm=norm)
axp=ax.scatter(kx[mezzival:],ky[mezzival:],c=vals[mezzival:],norm=norm,edgecolors='black')
cb = plt.colorbar(axp)
'''