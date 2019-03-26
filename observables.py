# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:17:41 2019

@author: Nicco
"""
import numpy as np

def number(vecs,Ltilde):
    num=0
    for i in range(Ltilde**2):
        num+=np.conj(vecs[Ltilde**2:,i])@vecs[Ltilde**2:,i]
    return num

def densityfork(i,vecs,Ltilde): #density operator when we swich in momentum base
    asc,ordin=indtocord(i,Ltilde)
    n=0
    for i in range(Ltilde**2):
        vi=vecs[Ltilde**2:,i]
        for k in range(Ltilde**2):
            vk=vecs[Ltilde**2:,k]
            n+=np.exp(1j*(kx[i]-kx[k])*asc+1j*(ky[i]-ky[k])*ordin)*(np.conj(vi)@vk)
    return n

def indtocord(i,Ltilde):
    cord=np.array([0,0])
    cord[0]=np.floor(i/Ltilde)
    cord[1]=i%Ltilde
    return cord

def cordtoind(asc,ordin,Ltilde):
    return ordin*Ltilde+asc

def density(i,vecs,Ltilde):
    return sum(np.conj(vecs[i+Ltilde**2,Ltilde**2:])*vecs[i+Ltilde**2,Ltilde**2:])


fig=plt.figure(1)
ax=fig.add_subplot(111, projection='3d')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_title('a='+str(a)+', t='+str(t)+', mu='+str(mu0)+', delta='+str(delta))
asc=np.linspace(0,Ltilde-1,Ltilde,dtype=np.int)
asc,ordin=np.meshgrid(asc,asc)    
ax.scatter(asc,ordin,density(cordtoind(asc,ordin,Ltilde),vecs,Ltilde),'bo')

cane=0
for i in range(Ltilde**2):
    #asc,ordin=indtocord(i,Ltilde)
    cane+=density(i,vecs,Ltilde)
print(cane*a**2)
print(number(vecs,Ltilde))
print(Ltilde**2)

