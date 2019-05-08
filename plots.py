# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:01:44 2019

@author: Nicco
"""
from __future__ import division
import matplotlib.pyplot as plt
import functions as fx
import numpy as np


'''
loading of data from files (if processon cluster)
'''
'''
L=8 
Ltilde=80
a=L/Ltilde

filedens=np.load('../plots/29aprile/density'+str(Ltilde)+'.npz')
dens=filedens['arr_0']/(a**2) #set vecs if file is local, dens if it's on cluster (in this case remember to divide by a**2)
filevals=np.load('../plots/29aprile/provals'+str(Ltilde)+'.npz')
vals=filevals['arr_0']
'''
'''
setup of the mesh
'''
asc=np.linspace(0,Ltilde-1,Ltilde,dtype=np.int) #edge of system with correct dimension
asc,ordin=np.meshgrid(asc,asc) #mesh of lattice grid
dens=fx.densityplot(asc,ordin,vecs,Ltilde)/(a**2) #density in each point (scaled), comment if it's on cluster

'''
calculation of observables
'''

#dens=np.zeros(Ltilde**2)
#for i in range(Ltilde**2):
#    dens[i]=sum(np.conj(vecs[i+Ltilde**2,(Ltilde**2):])*vecs[i+Ltilde**2,(Ltilde**2):])
#print('1')
'''
lop=fx.angumomop(Ltilde) #angular momentum operator using my discretization on lattice
lz=np.zeros(Ltilde**2)
for i in range(Ltilde**2):
    lz[i]=fx.expvalue(lop,vecs[Ltilde**2:,i+Ltilde**2])
print('2')

lopk=fx.angumomktop(Ltilde,L)  #angular momentum 
lzk=np.zeros(Ltilde**2)
for i in range(Ltilde**2):
    lzk[i]=fx.expvalue(lopk,vecs[Ltilde**2:,i+Ltilde**2])

plt.figure(1)
plt.plot(np.linspace(0,Ltilde**2,Ltilde**2),lzk,'bo')
plt.figure(2)
plt.plot(np.linspace(0,Ltilde**2,Ltilde**2),lz,'bo')
'''
num=sum(sum(dens))

fig=plt.figure(Ltilde)

'''
3D plot
'''
ax=fig.add_subplot(121, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Ltilde='+str(Ltilde)+', t='+str(t)+', mu='+str(mu)+', delta='+str(delta))
ax.scatter(asc*a,ordin*a,dens,'go') #plot of superfluid density (scaled edges)

'''
heat plot (on side of the 3d one)
'''
ax2=fig.add_subplot(122)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
axp=ax2.contourf(asc*a,ordin*a,dens,500)
cb = plt.colorbar(axp) #colorbar with correct scale

'''
energy spectrum visualized around zero
'''
plt.figure(3)
plt.grid(True)
plt.ylim(-100,100)
plt.xlim((len(vals)/2)-100,(len(vals)/2)+100)
plt.plot(np.linspace(0,len(vals),len(vals)),vals/(a**2),'ro',markersize=1.5)

'''
print of important quantities
'''

print("size="+str(Ltilde)) #lattice size
print("num="+str(num)) #number of particles
#print("Lz="+str(lztot)) #angular momentum (calculated using lattice discretization)
#print("Lzk="+str(sum(lzk))) #angular momentum (calculated using momentum representation)
print('num_teo='+str(((L**2)/(4*np.pi))*(mu/t)))
print('gap='+str(vals[Ltilde**2+2]-vals[Ltilde**2+1]))

