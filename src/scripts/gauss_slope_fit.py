import numpy as np
from astropy.table import Table
import paths
from astropy.modeling import models, fitting
from astropy.modeling.models import Polynomial1D
from astropy.modeling.models import Linear1D

import matplotlib.pyplot as plt
import warnings

# dummy file that generates the output for plot_ccf.py
#
# this should contain the CCF generating code
#
#
# output files should be ccf_data1.txt

import matplotlib as mpl

#mpl.use('macosx')

pp1 = models.Gaussian1D(100, 0, 10)
pp2 = models.Linear1D(slope=0.3, intercept=2.0)
model = pp1+pp2

rng = np.random.default_rng(seed=42)

x = np.linspace(-50,50,100)

y = pp1(x) + pp2(x) + rng.normal(0,5,x.shape)



# Now to fit the data create a new superposition with initial
# guesses for the parameters:
gg_init = models.Gaussian1D(50, 0, 5) + models.Linear1D(0,0)
fitter = fitting.SLSQPLSQFitter()



with warnings.catch_warnings():
    # Ignore a warning on clipping to bounds from the fitter
    warnings.filterwarnings('ignore', message='Values in x were outside bounds',
                            category=RuntimeWarning)
    gg_fit = fitter(gg_init, x, y)

print(gg_fit)

plt.plot(x, y, 'ko')
plt.plot(x, gg_fit(x))
plt.xlabel('Position')
plt.ylabel('Flux')

plt.show()
quit()

def gsv():

	return func


def gauss_slope_fit():
	'fit a gaussian plus a slope plus an offset'



from numpy.random import default_rng
rng = default_rng(5555) # seed number guarantees this is repeatable

vel = np.arange(-50,50,10)

N11 = np.ones_like(vel)*1 + rng.standard_normal(10)
N12 = np.ones_like(vel)*2 + rng.standard_normal(10)
N13 = np.ones_like(vel)*3 + rng.standard_normal(10)


t = Table([vel,N11,N12,N13],names=['vel', 'N11', 'N12', 'N13'])

t.write(paths.data / 'ccf_data1_mean.ecsv',format='ascii.ecsv',overwrite=True)
