# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:17:41 2019

@author: Nicco
"""

import numpy as np
import functions as fx

def number(vecs,Ltilde):
    num=0
    for i in range(Ltilde**2):
        num+=np.conj(vecs[Ltilde**2:,i])@vecs[Ltilde**2:,i]
    return num

def density(asc,ordin,vecs,Ltilde):
    i=fx.cordtoind(asc,ordin,Ltilde)
    return sum(np.conj(vecs[i+Ltilde**2,:Ltilde**2])*vecs[i+Ltilde**2,:Ltilde**2])

def densityplot(asc,ordin,vecs,Ltilde):
    n=np.zeros((Ltilde,Ltilde))
    for i in range(Ltilde):
        for j in range(Ltilde):
            n[i][j]=density(asc[i][j],ordin[i][j],vecs,Ltilde)
    return n
fig=plt.figure(1)
ax=fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('a='+str(a)+', t='+str(t)+', mu='+str(mu0)+', delta='+str(delta))
asc=np.linspace(0,Ltilde-1,Ltilde,dtype=np.int)
asc,ordin=np.meshgrid(asc,asc)
ax.set_zlim(-1,1)
ax.scatter(asc,ordin,densityplot(asc,ordin,vecs,Ltilde),'bo')

cane=sum(sum(densityplot(asc,ordin,vecs,Ltilde)))
print(cane)
print(number(vecs,Ltilde))
print(Ltilde**2)

