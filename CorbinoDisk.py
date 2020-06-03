# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:32:22 2020

@author: niccobal
"""
import numpy as np
from scipy import linalg as alg
from matplotlib import pyplot as plt
from scipy import special as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable

#lattice parameters
rmin=0.01 #inner radius
rmax=3.88 #outer radius
L=20 #radius discretization
N=50 #angle discretization
drad=(rmax-rmin)/L

#hamiltonian parameters
T=1 #hopping term
M=8. #chemical potential
D=3 #coupling term

############################################################################################
#definition of functions

#kinetic energy part
def corb_kinet(rrmin,rrmax,LL,NN,TT,MM):
    matr=np.zeros((NN*LL,NN*LL),dtype=np.complex128) #initialization of a square matrix
    drad=(rrmax-rrmin)/LL
    for i in range(LL*NN):
        l=np.floor(i/NN)                       #radial index of point i
        neigh=int((i+1)%NN+NN*np.floor(i/NN))  #right neighbour for angular coupling

        ang_coupl=-TT*(NN/(2.*np.pi))**2*(1./(rmin+l*drad))**2 #hopping in angular direction
        rad_coupl=TT*(rrmin+(l+0.5)*drad)/(drad)**2            #hopping in radial direction

        if ((i+NN)<(LL*NN)):
#            if(l==0):
#                matr[i,i+NN]=0
#                matr[i+NN,i+NN]+=0.5*rad_coupl/(rrmin+(l+1)*drad)
#            else:
            matr[i,i+NN]+=-rad_coupl/np.sqrt((rrmin+l*drad)*(rrmin+(l+1)*drad))
            matr[i+NN,i+NN]+=0.5*rad_coupl/(rrmin+(l+1)*drad)
            
        matr[i,neigh]+=ang_coupl
        matr[i,i]+=-ang_coupl-0.5*MM+0.5*rad_coupl/(rrmin+l*drad)
    matr=matr+matr.T.conj()
    return matr
 
#superconducting coupling
def corb_coupl(rrmin,rrmax,LL,NN,DD):
    matr=np.zeros((NN*LL,NN*LL),dtype=np.complex128) #initialization of a square matrix
    drad=(rrmax-rrmin)/LL
    for i in range(LL*NN):
        l=np.floor(i/NN)                       #radial index of point i
        n=i%NN                                 #angular index of point i
        neigh=int((i+1)%NN+NN*np.floor(i/NN))  #right neighbour for angular coupling
    
        ang_coupl=0.5*DD*(NN/(2*np.pi))*(1/(rrmin+l*drad))*np.exp(1j*((n/NN)+0.5)*(2*np.pi))  #supercond. coup. angular direction
        rad_coupl=0.5*DD*((rrmin+l*drad)/drad)*np.exp(1j*(n)*(2*np.pi)/NN)/np.sqrt((rrmin+l*drad)*(rrmin+(l+1)*drad)) #supercond. coup. radial direction
    
        if ((i+NN)<(LL*NN)):
#            if(l==0):
#                matr[i,i+NN]=0
#            else:
            matr[i,i+NN]=rad_coupl
        matr[i,neigh]=ang_coupl
    matr=matr-matr.T
    return matr
    
############################################################################################

#construction of hamiltonian
kinet=corb_kinet(rmin,rmax,L,N,T,M) #kinetic part
coup=corb_coupl(rmin,rmax,L,N,D)    #coupling part
ham=np.hstack((np.vstack((kinet,coup)),np.vstack((-(coup.conj()),-(kinet.T))))) #construction by stacking

coupq=np.matmul(coup,-coup.conj())
pha=np.triu(np.full((L*N,L*N),1j*10**(-7)),1)+np.triu(np.full((L*N,L*N),1j*10**(-7)),1).T.conj()
#coupq=coupq+pha
print(np.where(coupq-coupq.T.conj()!=0))

#exact diagonalization
vals,vecs=alg.eigh(ham)

############################################################################################

#colorplot of density 
punti=np.linspace(0,L*N-1,L*N) #initialization of a square grid to be recast as radial
l=np.floor(punti/N) #indices l,n of the square grid
n=punti%N
x=(rmin+l*drad)*np.cos(n*2*np.pi/N) #corresponding coordinates for the radial grid
y=(rmin+l*drad)*np.sin(n*2*np.pi/N)

dens=np.zeros(L*N) #density of superfluid 
for i in range(L*N):
    dens[i]=sum((vecs[i+L*N,(L*N):].conj()*vecs[i+L*N,(L*N):]))/((2*np.pi/N)*((rmin+l[i]*drad)*drad))

#dens=vecs[:,2].conj()*vecs[:,2]/((2*np.pi/N)*((rmin+l*drad)*drad)) #to use only when diagonalizing just free hamiltonian

#figure: density of superfluid
fig=plt.figure(L,figsize=(10,10))
fig.add_subplot(111,aspect='equal')

plt.scatter(x,y,c=dens,marker='o') #scatter plot of density

plt.grid(True)
plt.ylim(-rmax-0.5,rmax+0.5)
plt.xlim(-rmax-0.5,rmax+0.5)
plt.tight_layout()
plt.colorbar()


fig=plt.figure(4,figsize=(10,10)) #initialization of plot
ax2=fig.add_subplot(111) #plot inside figure
ax2.set_xlabel('x')
ax2.set_ylabel('y')
axp=ax2.scatter(x,y,c=dens) #contour plot of density

divider = make_axes_locatable(ax2) #setup of the colorbar on the top (next three lines)
cax = divider.append_axes('top',size="5%", pad=0.05)
plt.colorbar(axp,orientation='horizontal', cax=cax,ticks=np.linspace(0,np.max(dens),5))
cax.xaxis.set_ticks_position("top")
plt.tight_layout()


############################################################################################
#radial plot of laplacian eigenfunctions on the disk, to use only when diagonalizing just free hamiltonian
def radial(r,rmax,m,n): #radial component of the basis functions
    m=np.abs(m)
    N=(rmax**2/2)*(sp.jv(m+1,sp.jn_zeros(m,n)[n-1]))**2
    return (1./np.sqrt(N))*sp.jv(np.abs(m),r*sp.jn_zeros(m,n)[n-1]/rmax)

def basisfunc(r,phi,rmax,m,n): #multiplication by the proper phase
    return radial(r,rmax,m,n)*np.exp(1j*m*phi)*(1./np.sqrt(2*np.pi))

#radial component of density retrieved at a given radius from exact diagonalization
densrad=np.zeros(L) 
for i in range(L):
    densrad[i]=dens[(i)*N]
    
#plot of exact radial density vs retrieved from exact diagonalization
plt.figure(70)
asc=np.linspace(rmin,rmax,L)
plt.title('mu='+str(M)+', delta='+str(D)+', T='+str(T))
plt.xlabel('Radius')
plt.ylabel('Density')
plt.scatter(asc,densrad,s=5, )  #exact diagonalization density
#plt.scatter(asc,basisfunc(asc,0,rmax,0,1).conj()*basisfunc(asc,0,rmax,0,1)) #exact density from eigenfunctions

plt.grid(True)
plt.tight_layout()

#plot of energy spectrum around zero
plt.figure(3)
plt.grid(True)
plt.ylim(-50.5,50.5)
plt.ylabel('Energy')
plt.xlim((len(vals)/2)-25,(len(vals)/2)+25)
plt.xticks([], [])

plt.plot(np.linspace(0,len(vals),len(vals)),vals,'bo') #plot of energies
plt.tight_layout()

