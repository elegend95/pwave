# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:23:32 2019

@author: Nicco
this code contains all the useful routines to compute energy spectrum for PBC Bogoliubov Hamiltonian
"""

import Code1

#calculation of impulses of eigenvectors employing translational invariance of states
kx=np.zeros(len(vals))
ky=np.zeros(len(vals))
for r in range(len(vals)):
    kx[r]=-1j*np.log(vecs[0,r]/vecs[1,r])/a   
    ky[r]=-1j*np.log(vecs[Ltilde,r]/vecs[0,r])/a
    if (np.abs(vecs[0,r])<10**-8)&(np.abs(vecs[1,r])<10**-8): #correct behaviour when kx and ky should be zero (bit of workaround)
        kx[r]=-1j*np.log((vecs[Ltilde**2,r])/(vecs[1+Ltilde**2,r]))/a
    if (np.abs(vecs[Ltilde,r])<10**-8)&(np.abs(vecs[0,r])<10**-8):
        ky[r]=-1j*np.log((vecs[Ltilde+Ltilde**2,r])/(vecs[Ltilde**2,r]))/a

#3D surface figure, setting parameters
asc=np.linspace(-np.pi/a,np.pi/a,100)
asc,ordin=np.meshgrid(asc,asc)    
fig=plt.figure(1)
ax=fig.add_subplot(111, projection='3d')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_title('a='+str(a)+', t='+str(t)+', mu='+str(mu)+', delta='+str(delta))
ax.view_init(0, 90)
#sets the ticks on axis in multiples of py
ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))

#theoretical bogoliubov dispersion
ax.plot_surface(asc, ordin, fx.enbog(asc,ordin,t,a,mu,delta), alpha=0.2,color='b')
ax.plot_surface(asc, ordin, -fx.enbog(asc,ordin,t,a,mu,delta), alpha=0.2,color='b')
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

#heat plot of upper band of bogoliubov dispersion
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Ltilde='+str(Ltilde)+', t='+str(t)+', mu='+str(mu)+', delta='+str(delta))
ax.set_xlabel('kx')
ax.set_ylabel('ky')
asc=np.linspace(-np.pi/a,np.pi/a,100)
asc,ordin=np.meshgrid(asc,asc)
mezzival=Ltilde**2
ax.set_title('a='+str(a)+', t='+str(t)+', mu='+str(mu)+', delta='+str(delta))
mi = np.min((vals[mezzival:].min(), fx.enbog(asc,ordin,t,a,mu,delta).min()))
ma = np.max((vals[mezzival:].max(), fx.enbog(asc,ordin,t,a,mu,delta).max()))
norm = clr.Normalize(vmin=mi,vmax=ma)
ax.contourf(asc,ordin,fx.enbog(asc,ordin,t,a,mu,delta),200,norm=norm)
axp=ax.scatter(kx[mezzival:],ky[mezzival:],c=vals[mezzival:],norm=norm,edgecolors='black')
cb = plt.colorbar(axp)
     


