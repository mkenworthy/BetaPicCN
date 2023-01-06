# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:29:29 2022

@author: demooij
"""

import matplotlib.pyplot as plt
import numpy as np
import paths

data=np.loadtxt(paths.data/'results_HARPS.txt')

T=np.unique(data[:,0])
N=np.unique(data[:,1])

sig_sim=data[:,-1].reshape(np.size(T),np.size(N))

fig, ax = plt.subplots(1,1,figsize=(8,6))
pcm = ax.pcolormesh(N,T,sig_sim,shading='nearest')
ax.set_xscale("log")
ax.set_yscale("log")
fig.colorbar(pcm,ax=ax)
ax.set_xlabel('N [$cm^{-2}$]',fontsize=16)
ax.set_ylabel('T [K]',fontsize=16)
plt.savefig(paths.figures/'HARPS_CN_grid.pdf', bbox_inches='tight')
