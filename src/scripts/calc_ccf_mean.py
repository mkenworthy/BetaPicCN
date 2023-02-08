import numpy as np
from astropy.table import Table
import paths


# dummy file that generates the output for plot_ccf.py
#
# this should contain the CCF generating code
#
#
# output files should be ccf_data1.txt

from numpy.random import default_rng
rng = default_rng(5555) # seed number guarantees this is repeatable

vel = np.arange(-50,50,10)

N11 = np.ones_like(vel)*1 + rng.standard_normal(10)
N12 = np.ones_like(vel)*2 + rng.standard_normal(10)
N13 = np.ones_like(vel)*3 + rng.standard_normal(10)


t = Table([vel,N11,N12,N13],names=['vel', 'N11', 'N12', 'N13'])

t.write(paths.data / 'ccf_data1_mean.ecsv',format='ascii.ecsv',overwrite=True)