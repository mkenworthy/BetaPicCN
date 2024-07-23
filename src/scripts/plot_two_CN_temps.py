import matplotlib.pyplot as plt
import numpy as np
import paths

from analyse_HARPS_spectra import get_mdl,wavegrid


wlen,t1,_=get_mdl('CN/CN_{0:04d}K.npy'.format(30),N=1e12)
wlen,t2,_=get_mdl('CN/CN_{0:04d}K.npy'.format(2000),N=1e13)



fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_xlim(382,388.6)

ax.set_xlabel('Wavelength [$nm$]',fontsize=16)
ax.set_ylabel('Transmission',fontsize=16)

ax.plot(wlen/10,t1)
ax.plot(wlen/10,t2)
#plt.show()
plt.savefig(paths.figures / 'two_CN_temps.pdf', bbox_inches='tight')

