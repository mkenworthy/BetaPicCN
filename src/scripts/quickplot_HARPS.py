# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:29:29 2022

@author: demooij
"""

import matplotlib.pyplot as plt
import numpy as np

data=np.loadtxt('results.txt')

T=np.unique(data[:,0])
N=np.unique(data[:,1])
sig_sim=data[:,-1].reshape(2,1)
plt.pcolormesh(N,T,sig_sim)
plt.xscale("log")
plt.yscale("log")
plt.colorbar()
plt.xlabel('N [$cm^{-2}$]')
plt.ylabel('T [K]')
plt.savefig('sigfull.png')


plt.figure()
plt.pcolormesh(N,T,sig_sim,vmin=-10,vmax=50)
plt.xscale("log")
plt.yscale("log")
plt.colorbar()
plt.xlabel('N [$cm^{-2}$]')
plt.ylabel('T [K]')
plt.savefig('sigfocus.png')