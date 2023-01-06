# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:54:53 2022

@author: demooij
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
import paths

mpl.rc('image', interpolation='nearest', origin='lower', cmap = 'gray')

from pylab import rcParams
from scipy import interpolate
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip
from astropy.stats.sigma_clipping import sigma_clipped_stats
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial as P
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats as stats
import time
import gc

rcParams['figure.figsize'] = 12, 8

## this is to convolve an arbitrary spectrum in (wave, spec) with velocity broadening effects.
## dv is in km/s (which here we're setting to c/R)
def wavegrid(lmin,lmax,R):
    '''lmin is the minimum wavelength, lmax is the maximum wavelength you want, R is the spectral resolution
    returns a wavelength vector that is constant in velocity space.
    then interpolate to this wavelength grid'''
    dx=np.log(1.+1./R) ## the log10 gives you constant velocity spacing
    x=np.arange(np.log(lmin),np.log(lmax),dx)
    return np.exp(x)


## Read in the model from a file and convert the crossesction to opacity for a given columndensity.
def get_mdl(fn='CN/CN_0300K.npy',N=1e15,FWHM_kernel=3e5/1.1e5):
    
    ## set sampling for convolution to 0.25 km/s
    res_interp=12e5
    
    c = np.load(paths.data/fn)
    w = np.load(paths.data/'CN/CN_wavelengths.npy') #np.arange(3850, 3885, 0.0001)
    tau = N * c
    prof = np.exp(-tau)
    
      
    ## Sample the model at a constant resolution for convolution
    w_mdl=wavegrid(np.min(w)-1., np.max(w)+1.,res_interp)
    prof_i=interp1d(w,prof,kind='linear',bounds_error=False,fill_value=1.)
    mdl=prof_i(w_mdl)

    ## generate a Gaussian kernel with the FWHM at the instrumental resolution & convolve model
    ## *** TBD: Allow other line broadening (e.g. based on Ca lines or even for different resolutions) ***
    FWHM_kernel=3e5/1.10e5
    x_kernel=np.arange(-25.,25.001,(3e5/res_interp))
    kernel=np.exp( - ((x_kernel)**2)/2./(FWHM_kernel/2.35)**2)
    kernel=kernel/np.sum(kernel)
    mdl_conv=np.convolve(mdl,kernel,mode='same')
    idx=(w_mdl >= np.min(w))*(w_mdl <= np.max(w))

    ## set up interpolation object for the model to use in the CCF code
    imdl_conv=interp1d(w_mdl[idx],mdl_conv[idx],kind='linear',bounds_error=False,fill_value=1.)

    
    return w,prof,imdl_conv



##Read in data
def read_data(order=[0,1,2,3,4,5,6,7]):
    nspec = fits.getdata(paths.data/'2d_nspec.fits')
    spec = fits.getdata(paths.data/'2d_spec.fits')
    wlen = fits.getdata(paths.data/'2d_wavelength.fits')
    bary = fits.getdata(paths.data/'2d_baryvel.fits')
    return nspec,spec[:,order,:],wlen[order,:],bary


##Find strongest exocomets


## Estimate blaze
def simple_blaze(wlen,master_blaze,nskip,nbins,binsize):
    ww0=wlen[nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()
    co=master_blaze[nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()

    mf=np.nanmean(co,axis=1)
    ww=np.nanmean(ww0,axis=1)

    for it in range(nbins):
        f_tmp=co[it,:]
        l=np.polyfit(ww0[it,:],co[it,:]-mf[it],1)
        res=co[it,:]-mf[it] - np.polyval(l,ww0[it,:])
        sig=np.median(np.abs(res-np.median(res)))

        idx=np.abs(res/sig)>4.
        if np.sum(idx)>0:
            f_tmp[idx]=np.nan
        co[it,:]=f_tmp.copy()

    mf=np.nanmean(co,axis=1)
    ww=np.nanmean(ww0,axis=1)

    flux_int=interpolate.interp1d(ww,mf,kind='cubic',fill_value='extrapolate',bounds_error=False)
    interp_flux = flux_int(wlen)

    return interp_flux


## Higher order blaze variation corrections
def blaze_var_cor(wlen, cor,nskip,nbins,binsize,n_frm,poly_ord):
    for frm in range(n_frm):
        ww0=wlen[nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()
        co=cor[frm,nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()
        mf=np.nanmean(co,axis=1)
        ww=np.nanmean(ww0,axis=1)

        for it in range(nbins):
            f_tmp=co[it,:]
            res=co[it,:]-mf[it]
            sig=np.median(np.abs(res-np.median(res)))
            idx=np.abs(res/sig)>4.
            if np.sum(idx)>0:
                f_tmp[idx]=np.nan
            co[it,:]=f_tmp.copy()

        mf=np.nanmean(co,axis=1)
        ww=np.nanmean(ww0,axis=1)

        l=np.polyfit(ww,mf,poly_ord)
        cor[frm,:]=cor[frm,:]/np.polyval(l,wlen)

    return cor


## Correct overall blaze etc. using a spline interpolation + binning, this is done iteratively
## for N_iter times to allow any residual blaze-corrections to be performed.
## This needs to be run for an individual order (given by ord)
def correct_spec_blaze_and_star_init(wlen,spec,ord=0,binsize=80,nskip=20,binsize2=400, N_iter=5,poly_ord=2,use_bounds=True):

    # Butterworth filter parameters
    fs=1.
    v_cutoff=125. # high pass filter (minimum velocity below which signals get through)
    b_order=5
    normal_cutoff=(1./v_cutoff)/(0.5*fs) # 1/velocity becasue we're in frequency space, and 0.5 is Nyquist sampling related...

    # setting bounds for certain orders because they don't extend fully across the mid-point of order overlap
    if use_bounds:
        if ord==0:
            lbound=220
            ubound=3680
        else:
            lbound=0
            ubound=spec.shape[2]
    else:
        lbound=0
        ubound=spec.shape[2]

    cor=spec[:,ord,:].copy()

    # there's one order where fluxes occasionally go negative. Flag and remove.
    idx=(cor<=0)
    cor[idx]=1.

    cor2=cor.copy()

    n_frm,n_ord,n_pix=spec[:,:,lbound:ubound].shape


    # calculate integer number of bins for the two different bin sizes
    nbins=(n_pix-2*nskip)//binsize
    nbins2=(n_pix-2*nskip)//binsize2

    # loop over the number of iterative clips
    for i in range(N_iter):

        master_blaze=np.median(cor,axis=0)
        interp_flux=simple_blaze(wlen[ord,lbound:ubound],master_blaze[lbound:ubound],nskip,nbins,binsize)
        cor[:,lbound:ubound]=cor[:,lbound:ubound]/interp_flux[np.newaxis,:]
        cor2[:,lbound:ubound]=blaze_var_cor(wlen[ord,lbound:ubound], cor[:,lbound:ubound],nskip, nbins2, binsize2,n_frm,poly_ord)
        cor=cor2.copy()


    # Butterworth filter removes the stellar lines
    b, a = signal.butter(b_order, normal_cutoff, btype = "highpass", analog = False)
    if (ord >1) and (ord<5):
        for i in range(cor.shape[0]):
            y = signal.filtfilt(b, a, cor[i,:])
            cor[i,:]=y.copy()

    return cor

## Find the comets in the Ca II H line
def find_comets(wlen,spec_m,ord_FEB=7,use_all=True):
    #Select the order with Ca II H line
    wlen_FEB = wlen[ord_FEB]
    spec_FEB = np.transpose(spec_m[:,ord_FEB,:])
    
    #remove outliers to get a good sigma-clipped spectrum for stationary features
    filtered_data = sigma_clip(spec_FEB, sigma_lower=1.0, sigma_upper=5, maxiters=5, axis=1, copy=True, masked=True)
    spec_Ca_only = np.ma.mean(filtered_data, axis=1)
    spec_FEB_norm = spec_FEB/spec_Ca_only[:,np.newaxis]
    
    # look between 3968.85 and 3971.0, find the minima (=maximum absorption)
    # range chosen to avoid most of the core. 
    # use_all will look for blueshifted comets too
    if use_all:
        m_p = (wlen_FEB > 3968.85) * (wlen_FEB < 3971.0)  + (wlen_FEB > 3966.0) * (wlen_FEB < 3968.65)
    else:
        m_p = (wlen_FEB > 3968.85) * (wlen_FEB < 3971.0)
    wlen_peak = wlen_FEB[m_p]
    spec_peak = 1. - (spec_FEB_norm[m_p])
    
    wheremax = np.argmax(spec_peak, axis=0)
    ampl_FEB_clip = np.max(spec_peak, axis=0)
    wlen_FEB_clip = wlen_peak[wheremax]

    # we now have the heights of the peaks in what_max, and the positions of the peaks in wlen_max
    # we make two separate sets to make: FEBs and no FEBs
    # 0.5 means 50% strength for pickup as an FEB
    # 0.1 means less than 10% strength for no FEB
    FEB_sel =   (ampl_FEB_clip > 0.5)
    star_sel =  (ampl_FEB_clip < 0.1)

    return FEB_sel,star_sel,wlen_FEB_clip

   



##Set up a simple CCF code. N.B. This is the unweighted CCF (weighting currently commented out...)
def c_correlate(l_data,data,imodel,dv=np.linspace(-50,50,101)):
    npix=dv.shape[0]
    ccf=np.zeros((data.shape[0],npix))
    ccf0=np.zeros((data.shape[0],npix))
    dat=data-np.mean(data,axis=1)[:,np.newaxis]

    
    z0=1.-dv/3e5
    for i,z in enumerate(z0):
        mdl=imodel(l_data*z)
        mdl=mdl-np.mean(mdl)
        ccf[:,i]=np.sum( data*mdl[np.newaxis,:],axis=1)#/np.sqrt(np.sum(data**2,axis=1)*np.sum(mdl**2))
        ccf0[:,i]=ccf[:,i].copy()


    return ccf,ccf0


## Inject the model into the data 
##                 *** TBD ***                   
## Inject at a much earlier stage of the process 
def inject_mdl(l_data,data,imodel,dv=np.zeros(51)):
    sim_dat=data.copy()
    z0=1.-dv/3e5
    for i in range(data.shape[0]):
        mdl=imodel(l_data*z0[i])
        sim_dat[i,:]=data[i,:]*mdl
    return sim_dat



def generate_CCFs_orders(wlens,data_for_ccf,spec,broadening=10.,binsize=80,binsize2=400,nskip=20,T_gas=10,N=1e12,v_cutoff=50,v_comet=0.):

    ## Set order start and end pixels to exclude bad edges (need to optimize!)
    o_start=np.array([ 225,  15,  26,  26,  26,  26])
    o_end  =np.array([3670,4050,4060,4060,4060,2500])

    ## define arrays to store ccfs 
    v_ccf=np.arange(-250.,250.,1.)
    m_ccf=np.zeros((data_for_ccf.shape[0],wlens.shape[0],v_ccf.shape[0]))
    m_sim_ccf=np.zeros((data_for_ccf.shape[0],wlens.shape[0],v_ccf.shape[0]))

    
    for ord in range(wlens.shape[0]):
        
        ## get the opacities and convert to line-profile
        w,prof,imdl_conv=get_mdl('CN/CN_{0:04d}K.npy'.format(T_gas),N=N,FWHM_kernel=3e5/1.1e5)
        w_sim,prof_sim,imdl_conv_sim=get_mdl('CN/CN_{0:04d}K.npy'.format(T_gas),N=N,FWHM_kernel=broadening)

        ## Calculate CCF for data
        ccf,ccf0=c_correlate(wlens[ord,o_start[ord]:o_end[ord]],data_for_ccf[:,ord,o_start[ord]:o_end[ord]],imdl_conv,dv=v_ccf)
        m_ccf[:,ord,:]=ccf.copy()

        
        ## Inject model into data 
        sim_dat=spec.copy() 
        sim_dat[:,ord,:]=inject_mdl(wlens[ord,:],spec[:,ord,:],imdl_conv_sim,dv=v_comet)# np.zeros(spec.shape[0]))  
        sim_dat[:,ord,:]=correct_spec_blaze_and_star_init(wlens,sim_dat,ord,binsize=binsize,nskip=nskip,binsize2=binsize2, N_iter=5)
        sim_dat=np.vstack((sim_dat,np.mean(sim_dat,axis=0)[np.newaxis,:,:]))

        # cross correlate simulated spectra
        sim_ccf,sim_ccf0=c_correlate(wlens[ord,o_start[ord]:o_end[ord]],sim_dat[:,ord,o_start[ord]:o_end[ord]],imdl_conv,dv=v_ccf)
        m_sim_ccf[:,ord,:]=sim_ccf.copy()

    return v_ccf,m_ccf,m_sim_ccf



def gauss(x,A,mu,sig,offset):
    return(offset+A*np.exp( - (x-mu)**2/2./sig**2))

def gen_gauss(v_c):
    def n_gauss(x,A,sig,offset):
        return(offset+A*np.exp( - (x-v_c)**2/2./sig**2))
    return n_gauss


def estimate_significance(v_ccf,ccf,v_c=0.,dv=5.,T_gas=10.,N=1e12,ord=3,sim=False):
    # 15 km/s from the line central position is the outside limit
    v_outer = 15

    idx_outside=(np.abs(v_ccf-v_c)>dv)*(np.abs(v_ccf-v_c)<v_outer)
    idx_both=(np.abs(v_ccf-v_c)<v_outer)
    idx_line=np.abs(v_ccf-v_c)<dv
    if len(ccf.shape)==1:
        l=np.polyfit(v_ccf[idx_outside],ccf[idx_outside],1)
        mdl=np.polyval(l,v_ccf)
        rms=np.std(ccf[idx_outside]/mdl[idx_outside])
        popt,pcov=curve_fit(gauss,v_ccf[idx_both],ccf[idx_both]/mdl[idx_both],p0=[0.,v_c,dv/2.35,0.],sigma=rms*np.ones(v_ccf[idx_both].shape))
        perr=np.sqrt(np.diag(pcov))
    else:
        
        fit=np.zeros((ccf.shape[0]+1,3))
        err=np.zeros((ccf.shape[0]+1,3))
        
        #f,ax=plt.subplots(1)
        ccf_cor=ccf.copy()

        for i in range(ccf.shape[0]): # fit linear function in background region
            l=np.polyfit(v_ccf[idx_outside],ccf[i,idx_outside],1)
            mdl=np.polyval(l,v_ccf)

            # use the rms of the resiudal to get an error estimate
            rms=np.std(ccf[i,idx_outside]/mdl[idx_outside])
            ccf_cor[i,:]=ccf[i,:]-mdl
            try:
                popt,pcov=curve_fit(gen_gauss(v_c),v_ccf[idx_both],ccf[i,idx_both]-mdl[idx_both],p0=[0.,dv/2.35,0.],sigma=rms*np.ones(v_ccf[idx_both].shape))
                perr=np.sqrt(np.diag(pcov))              
                fit[i,:]=popt
                err[i,:]=perr
            except:
                print("no convergence")    
            #ax.plot(v_ccf[idx_both],ccf[i,idx_both]-mdl[idx_both])
            
        # taking the weighted mean of all CCFs     
        rms_o=np.std(ccf_cor[:,idx_outside],axis=1)
        w=1./rms_o**2
        m_ccf=np.sum(w[:,np.newaxis]*ccf_cor,axis=0)/np.sum(w)
        sig=1./np.sqrt(np.sum(w))

        # refit again using sig as the error on the weighted mean
        try:
            popt,pcov=curve_fit(gen_gauss(v_c),v_ccf[idx_both],m_ccf[idx_both],p0=[0.,dv/2.35,0.],sigma=sig*np.ones(v_ccf[idx_both].shape))
            perr=np.sqrt(np.diag(pcov))              
            fit[-1,:]=popt
            err[-1,:]=perr
        except:
            print("comb, no convergence")    
        return fit,err



def fold_comet(v_ccf,ccf,sim_ccf,T_gas,N,binsize,idx_comets,v_comets):

    ccf_comets=ccf
    sim_ccf_comets=sim_ccf

    # velocity range over which we try to line up cometa
    v_phasefold=np.arange(-60,60.1)
    
    # output arrays
    phasefold=np.zeros((ccf.shape[0],ccf.shape[1],v_phasefold.shape[0]))
    sim_phasefold=np.zeros((ccf.shape[0],ccf.shape[1],v_phasefold.shape[0]))

    for ord in range(ccf.shape[1]): # all orders...
        for i in range(phasefold.shape[0]): # for each night...
            ccf_i=interp1d(v_ccf,ccf[i,ord,:],kind='linear',bounds_error=False,fill_value=np.nan)
            sim_ccf_i=interp1d(v_ccf,sim_ccf[i,ord,:],kind='linear',bounds_error=False,fill_value=np.nan)
            phasefold[i,ord,:]=ccf_i(v_phasefold+v_comets[i])
            sim_phasefold[i,ord,:]=sim_ccf_i(v_phasefold+v_comets[i])

        if ord == 3: # the CN spectral order
            fit_sim,err_sim=estimate_significance(v_phasefold,sim_phasefold[:,ord,:],v_c=0.,dv=5.,T_gas=T_gas,N=N,sim=True)
            fit,err=estimate_significance(v_phasefold,phasefold[:,ord,:],v_c=0.,dv=5.,T_gas=T_gas,N=N)
    
    return v_phasefold,phasefold,sim_phasefold,fit,err,fit_sim,err_sim




## run ccf for a range of temperatures
def run_ccf_ord_multi_temp(wlens,data_for_ccf,spec,f,binsize=81,binsize2=400,nskip=25,v_cutoff=50,v_comet=0.,idx_comet=0.):
    ## Define a range of column densities to use (mainly injection)
    N0=np.hstack([np.arange(1,10)*1e11,np.arange(1,10)*1e12,np.arange(1,6)*1e13])
    N0=np.hstack([np.arange(1,4)*1e13])

    T_gas0=np.asarray([10, 20, 50, 100, 200, 300, 1000, 2000])
    T_gas0=np.asarray([10, 20,200])
    ntot = np.size(N0)*np.size(T_gas0)
    count = 1
    for T_gas in T_gas0:
        for N in N0:
            print(f'Running model {count} of {ntot} models for N0={N} and T_gas={T_gas}...')
            count = count + 1
            v_ccf,ccf,sim_ccf=generate_CCFs_orders(wlens,data_for_ccf,spec,binsize=binsize,binsize2=binsize2,nskip=nskip,T_gas=T_gas,N=N,v_cutoff=v_cutoff,v_comet=v_comet)
            
            v_phase,phase,sim_phase,fit,err,fit_sim,err_sim=fold_comet(v_ccf,ccf[:-1],sim_ccf[:-1],T_gas,N,binsize,idx_comet,v_comet)

            f.write("{0:.0f}  {1:.1e} {2:d}  {3:.2f}  {4:.2f}\n".format(T_gas, N, binsize, fit[-1,0]/err[-1,0],fit_sim[-1,0]/err_sim[-1,0]) )

'''
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
'''

CN_ord=3

# two binsizes - one for small scale and one for larger scales... determined empirically
binsize=81 # in pixels

nskip=25 # how many pixels to skip at the start of an echelle order to avoid edge effects

binsize2=400 # in pixels...

# read it all in
nspec,spec,wlen,bary=read_data()

# cor is corrected for blaze funtion and stellar lines
cor=spec.copy()
for ord in range(spec.shape[1]): # over all the orders...
    cor[:,ord,:]=correct_spec_blaze_and_star_init(wlen,spec,ord,binsize=binsize,nskip=20,binsize2=binsize2, N_iter=5)

## Find the strongest comet in the Ca II H line for each of the spectra
FEB_idx,star_idx,FEB_wave=find_comets(wlen,cor,ord_FEB=7,use_all=True)

# rest wavelength for CaII H line in the stellar rest frame
lam0=3968.469*(1.+20./3e5)

# converting the FEB features into a relative velocity for the comet
v_comet=3e5*(FEB_wave-lam0)/lam0

# stack both the mean spectrum and individual spectra for orders 0 to 6 (cut down on time and memory)
data_for_ccf=np.vstack((cor[:,0:6,:],np.mean(cor[:,0:6,:],axis=0)[np.newaxis,:,:]))
wlens=wlen[0:6,:]

f = open(paths.data/"results_HARPS.txt", "w")
run_ccf_ord_multi_temp(wlens,data_for_ccf,spec,f,binsize=binsize,binsize2=binsize2,nskip=nskip,v_cutoff=50,v_comet=v_comet,idx_comet=FEB_idx)
f.close()

