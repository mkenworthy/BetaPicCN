import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import paths

vgrid=fits.getdata(paths.data/'FEB_vgrid.fits')
orig_spec=fits.getdata(paths.data/'FEB_spec_orig.fits')
FEB_spec=fits.getdata(paths.data/'FEB_spec.fits')
FEB_idx=fits.getdata(paths.data/'FEB_idx.fits')
NO_FEB_idx=fits.getdata(paths.data/'NO_FEB_idx.fits')
FEB_depths=fits.getdata(paths.data/'FEB_depths.fits')
FEB_vel=fits.getdata(paths.data/'FEB_velocities.fits')


spec_FEB=FEB_spec[:,FEB_idx]
spec_NO_FEB=FEB_spec[:,NO_FEB_idx]



v_idx=(np.abs(vgrid)>8)*(np.abs(vgrid)<=200)

f,ax=plt.subplots(2,1,sharex=True)
ax[0].plot(vgrid,(orig_spec[:,FEB_idx])[:,1])
ax[0].plot(vgrid,np.mean(orig_spec[:,NO_FEB_idx],axis=1))
ax[1].plot(vgrid[v_idx],(spec_FEB[v_idx,:])[:,1])
ax[1].vlines((FEB_vel[FEB_idx])[1],0,2)
ax[1].plot(vgrid[v_idx],np.mean(spec_NO_FEB[v_idx,:],axis=1))
ax[0].set_xlim(-200,200)
ax[0].set_ylim(0,1.1)
plt.show()


f,ax=plt.subplots()
ax.plot(FEB_vel,FEB_depths,'k.')
plt.show()
