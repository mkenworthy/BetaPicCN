# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:29:29 2022

@author: demooij
"""

import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.io import fits
import matplotlib as mpl

mpl.use('macosx')


T0=fits.getdata(paths.data/'T_gas.fits')
N0=fits.getdata(paths.data/'Nf.fits')
w=fits.getdata(paths.data/'broadening.fits')

midpts_T=(T0[1:]+T0[:-1])/2.
x=np.arange(midpts_T.shape[0])+1
#T=10**np.polyval(np.polyfit(x,np.log10(midpts_T),1),np.arange(T0.shape[0]+1))
T=np.hstack((0.,midpts_T,1.5*T0[-1]))

midpts_N=(N0[1:]+N0[:-1])/2.
x=np.arange(midpts_N.shape[0])+1
N=10**np.polyval(np.polyfit(x,np.log10(midpts_N),1),np.arange(N0.shape[0]+1))



fit=fits.getdata(paths.data/'fit_cube_comet.fits')
sig0=fits.getdata(paths.data/'fit_err_cube_comet.fits')

mean=fit[:,:,:,-1,:]
sig=sig0[:,:,:,-1,:]#np.std(fit[:,:,:,:-1,:],axis=-2)/np.sqrt(fit.shape[-2])
SN=mean/sig
print(SN.shape,T.shape,N.shape,w.shape)

f,ax=plt.subplots(1,6,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
vmin=np.min(SN[:,:,:,0])
vmax=np.max(SN[:,:,:,0])
for i in range(5):
    im=ax[i].pcolormesh(N,T,SN[:,:,i,0],vmin=vmin,vmax=vmax,shading='flat')
    ax[i].set_xscale('log')
f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=12)
for i in range(5):
    ax[i].set_xlabel('N$\cdot$f [cm$^{-2}$]',fontsize=12)
    
f.tight_layout()




##run for simulation

fit=fits.getdata(paths.data/'sim_fit_cube_comet.fits')

mean=fit[:,:,:,-1,:]#np.mean(fit,axis=-2)
sig=np.std(fit[:,:,:,:-1,:],axis=-2)/np.sqrt(fit.shape[-2])



fit=fits.getdata(paths.data/'sim_fit_cube_comet.fits')
sig0=fits.getdata(paths.data/'sim_fit_err_cube_comet.fits')

mean=fit[:,:,:,-1,:]
sig=sig0[:,:,:,-1,:]
SN=mean/sig


f,ax=plt.subplots(1,6,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
vmin=np.min(SN[:,:,:,0])
vmax=np.max(SN[:,:,:,0])
for i in range(5):
    im=ax[i].pcolormesh(N,T,SN[:,:,i,0],vmin=vmin,vmax=vmax,shading='flat')
    ax[i].set_xscale('log')
f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=12)
for i in range(5):
    ax[i].set_xlabel('N$\cdot$f [cm$^{-2}$]',fontsize=12)
    
f.tight_layout()


##run for simulation -- 5-sigma


f,ax=plt.subplots(1,6,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
vmin=0
vmax=6
for i in range(5):
    im=ax[i].pcolormesh(N,T,SN[:,:,i,0],vmin=vmin,vmax=vmax,shading='flat')
    ax[i].set_xscale('log')
f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=12)
for i in range(5):
    ax[i].set_xlabel('N$\cdot$f [cm$^{-2}$]',fontsize=12)
    
f.tight_layout()



#####NOW STELLAR RESTFRAME####



fit=fits.getdata(paths.data/'fit_cube_stellar.fits')
sig0=fits.getdata(paths.data/'fit_err_cube_stellar.fits')

mean=fit[:,:,:,-1,:]
sig=sig0[:,:,:,-1,:]
SN=mean/sig




f,ax=plt.subplots(1,6,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
vmin=np.min(SN[:,:,:,0])
vmax=np.max(SN[:,:,:,0])
for i in range(5):
    im=ax[i].pcolormesh(N,T,SN[:,:,i,0],vmin=vmin,vmax=vmax,shading='flat')
    ax[i].set_xscale('log')
f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=12)
for i in range(5):
    ax[i].set_xlabel('N$\cdot$f [cm$^{-2}$]',fontsize=12)
    
f.tight_layout()




##run for simulation

fit=fits.getdata(paths.data/'sim_fit_cube_stellar.fits')
sig0=fits.getdata(paths.data/'sim_fit_err_cube_stellar.fits')

mean=fit[:,:,:,-1,:]
sig=sig0[:,:,:,-1,:]
##sig=np.std(fit[:,:,:,:-1,:],axis=-2)/np.sqrt(fit.shape[-2])
SN=mean/sig


f,ax=plt.subplots(1,6,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
vmin=np.min(SN[:,:,:,0])
vmax=np.max(SN[:,:,:,0])
for i in range(5):
    im=ax[i].pcolormesh(N,T,SN[:,:,i,0],vmin=vmin,vmax=vmax,shading='flat')
    ax[i].set_xscale('log')
f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=12)
for i in range(5):
    ax[i].set_xlabel('N$\cdot$f [cm$^{-2}$]',fontsize=12)
    
f.tight_layout()


##run for simulation -- 5-sigma


f,ax=plt.subplots(1,6,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
vmin=0
vmax=6
for i in range(5):
    im=ax[i].pcolormesh(N,T,SN[:,:,i,0],vmin=vmin,vmax=vmax,shading='flat')
    ax[i].set_xscale('log')
f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=12)
for i in range(5):
    ax[i].set_xlabel('N$\cdot$f [cm$^{-2}$]',fontsize=12)
    
f.tight_layout()




plt.show()








'''
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
'''
