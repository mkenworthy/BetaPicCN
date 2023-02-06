import matplotlib.pyplot as plt
import numpy as np
import paths


wlen=np.load(paths.data/'CN/CN_wavelengths.npy')

t1=np.load(paths.data/'CN/CN_0100K.npy')
t2=np.load(paths.data/'CN/CN_1000K.npy')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.set_yscale("log")

ax.set_xlim(382,386)
ax.set_ylim(1e-31,1e-15)

ax.set_xlabel('Wavelength [$nm$]',fontsize=16)
ax.set_ylabel('Transmission',fontsize=16)

ax.plot(wlen/10,t1)
ax.plot(wlen/10,t2)
#plt.show()
plt.savefig(paths.figures / 'two_CN_temps.pdf', bbox_inches='tight')
