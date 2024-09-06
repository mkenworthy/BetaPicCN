import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.io import fits

N0=fits.getdata(paths.data/"Nf.fits")
T_gas0=fits.getdata(paths.data/"T_gas.fits")
broadening0=fits.getdata(paths.data/"broadening.fits")
FEB_idx=fits.getdata(paths.data/"FEB_idx.fits")


##Find indices for 30K and 2000K
idx1=np.nonzero(T_gas0 == 30)
idx2=np.nonzero(T_gas0 == 2000)

##Find index for N = 10^12 cm^-2
idxN11=np.nonzero(N0 == 1e11)
idxN13=np.nonzero(N0 == 1e13)
idxN15=np.nonzero(N0 == 1e15)

##Index for v_broad=10km/s
idxW=np.nonzero(broadening0 == 10.)
print(idx1,idxN11,idxW)

v_ccf=fits.getdata(paths.data/"v_phasefold.fits")
ccf=fits.getdata(paths.data/"phase_cube.fits")
inj_ccf=fits.getdata(paths.data/"sim_phase_cube.fits")
ccf=ccf[:,:,:,FEB_idx,:]-np.median(ccf,axis=-1)[:,:,:,FEB_idx,np.newaxis]
inj_ccf=inj_ccf[:,:,:,FEB_idx,:]-np.median(inj_ccf,axis=-1)[:,:,:,FEB_idx,np.newaxis]
    



idx_outside=(np.abs(v_ccf)>25)*(np.abs(v_ccf<50))

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.set_xlim(-150,150)

ax.set_xlabel('Velocity [$km/s$]',fontsize=16)
ax.set_ylabel('CCF [a.u.]',fontsize=16)


ax.plot(v_ccf,np.mean(ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.max(np.mean(ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2))),label="T=30K, N=$10^{13}$")
ax.plot(v_ccf,np.mean(ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.max(np.mean(ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2))),label="T=3000K, N=$10^{15}$")
ax.plot(v_ccf,np.mean(inj_ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.max(np.mean(inj_ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2))),'--',label="Injected: T=30K, N=$10^{13}$")
ax.plot(v_ccf,np.mean(inj_ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.max(np.mean(inj_ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2))),'--',label="Injected: T=3000K, N=$10^{15}$")
ax.legend()
plt.draw()

plt.tight_layout()

fig.savefig(paths.figures/'ccf_mean_maxnorm.pdf')


fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.set_xlim(-150,150)

ax.set_xlabel('Velocity [$km/s$]',fontsize=16)
ax.set_ylabel('CCF [a.u.]',fontsize=16)


ax.plot(v_ccf,np.mean(ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.std( (np.mean(ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]))[idx_outside])),label="T=30K, N=$10^{13}$")
ax.plot(v_ccf,np.mean(ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.std( (np.mean(ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]))[idx_outside])),label="T=3000K, N=$10^{15}$")
ax.plot(v_ccf,np.mean(inj_ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.std( (np.mean(inj_ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]))[idx_outside])),'--',label="Injected: T=30K, N=$10^{13}$")
ax.plot(v_ccf,np.mean(inj_ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0])/np.abs(np.std( (np.mean(inj_ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]))[idx_outside])),'--',label="Injected: T=3000K, N=$10^{15}$")
ax.legend()
plt.draw()

plt.tight_layout()

fig.savefig(paths.figures/'ccf_mean_signorm.pdf')






fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.set_xlim(-150,150)

ax.set_xlabel('Velocity [$km/s$]',fontsize=16)
ax.set_ylabel('CCF [a.u.]',fontsize=16)

ax.plot(v_ccf,np.mean(ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]),label="T=30K, N=$10^{13}$")
ax.plot(v_ccf,np.mean(ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]),label="T=3000K, N=$10^{15}$")
ax.plot(v_ccf,np.mean(inj_ccf[idx1[0],idxN13[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]),'--',label="Injected: T=30K, N=$10^{13}$")
ax.plot(v_ccf,np.mean(inj_ccf[idx2[0],idxN15[0],idxW,:,:],axis=-2).reshape(v_ccf.shape[0]),'--',label="Injected: T=3000K, N=$10^{15}$")
ax.legend()
plt.draw()

plt.tight_layout()

fig.savefig(paths.figures/'ccf_mean.pdf')
plt.show()
